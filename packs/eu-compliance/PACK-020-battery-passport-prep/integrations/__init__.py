# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Integration Layer
===========================================================

Integration layer for the Battery Passport Prep Pack that provides
9-phase pipeline orchestration for EU Battery Regulation compliance,
MRV emissions routing for carbon footprint (Art 7), CSRD/ESRS E1/E2/E5
mapping, supply chain DD for critical minerals (Art 39-42), EUDR rubber
sourcing validation, EU Taxonomy DNSH for battery manufacturing (Activity
3.4), CSDDD adverse impact mapping, DATA agent routing for passport
fields (Art 77), 20-category health verification, and battery-category
setup wizard.

Components:
    - BatteryPassportOrchestrator: 9-phase pipeline with DAG dependency
      resolution, parallel phase execution, retry with exponential backoff,
      full and quick assessment modes, and SHA-256 provenance tracking
    - MRVBridge: Routes emissions from 11 battery-relevant MRV agents for
      Scope 1/2/3 manufacturing emissions and carbon intensity calculation
    - CSRDPackBridge: Maps 12 ESRS DRs (E1, E2, E5) to Battery Regulation
      articles for carbon footprint, pollution, and recycled content
    - SupplyChainBridge: Links supply chain agents for mineral traceability,
      CAHRA assessment, and OECD-aligned risk scoring for cobalt/lithium/nickel
    - EUDRBridge: Validates rubber and wood sourcing against EUDR deforestation
      requirements with country benchmarking and satellite verification
    - TaxonomyBridge: EU Taxonomy DNSH validation for Activity 3.4 (battery
      manufacture) with TSC cross-references to Battery Regulation
    - CSDDDBridge: Maps CSDDD Art 5-14 adverse impact findings to Battery Reg
      Art 39-42 supply chain DD with mineral-specific risk profiles
    - DataBridge: Routes battery passport data through 7 DATA agents for BOM,
      test results, supplier questionnaires, and quality validation
    - BatteryPassportHealthCheck: 20-category system health verification
    - BatteryPassportSetupWizard: Category-aware configuration setup wizard

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-020 Battery Passport <-- Composition <-- Zero Duplication
                              |
                              v
    PACK-017 (ESRS) <-- PACK-019 (CSDDD) <-- PACK-008 (Taxonomy)
    MRV Agents (11) <-- DATA Agents (7) <-- EUDR Agents

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__pack_id__ = "PACK-020"
__pack_name__ = "Battery Passport Prep Pack"

_loaded_integrations: List[str] = []

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        BatteryPassportOrchestrator,
        BatteryCategory,
        BatteryPipelinePhase,
        ExecutionStatus,
        OrchestratorConfig,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PARALLEL_PHASE_GROUPS,
        QUICK_ASSESSMENT_PHASES,
        PhaseProvenance,
        PhaseResult,
        PipelineResult,
        RetryConfig,
        REGULATION_ARTICLES,
    )
    _loaded_integrations.append("BatteryPassportOrchestrator")
except ImportError as e:
    BatteryPassportOrchestrator = None  # type: ignore[assignment,misc]
    logger.debug("BatteryPassportOrchestrator not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVBridge,
        MRVBridgeConfig,
        MRVAgentMapping as MRVBridgeAgentMapping,
        MRVScope,
        LifecycleStage,
        AgentStatus as MRVAgentStatus,
        ScopeEmissionsResult,
        ManufacturingEmissionsResult,
        CarbonIntensityResult,
        BATTERY_MRV_ROUTING,
        PERFORMANCE_CLASS_THRESHOLDS,
    )
    _loaded_integrations.append("MRVBridge")
except ImportError as e:
    MRVBridge = None  # type: ignore[assignment,misc]
    logger.debug("MRVBridge not available: %s", e)

# ---------------------------------------------------------------------------
# CSRD Pack Bridge
# ---------------------------------------------------------------------------
try:
    from .csrd_pack_bridge import (
        CSRDPackBridge,
        CSRDBridgeConfig,
        ESRSStandard,
        ESRSDisclosureMapping,
        ESRSImportResult,
        MappingRelevance,
        ImportStatus,
        ESRS_BATTERY_MAPPINGS,
    )
    _loaded_integrations.append("CSRDPackBridge")
except ImportError as e:
    CSRDPackBridge = None  # type: ignore[assignment,misc]
    logger.debug("CSRDPackBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Supply Chain Bridge
# ---------------------------------------------------------------------------
try:
    from .supply_chain_bridge import (
        SupplyChainBridge,
        SupplyChainBridgeConfig,
        SupplierTier,
        CriticalMineral,
        RiskLevel,
        DDComplianceStatus,
        SupplierRecord,
        SupplierDataResult,
        MineralSupplyChainResult,
        SupplierRiskResult,
        TierBreakdownResult,
        CAHRA_COUNTRIES,
    )
    _loaded_integrations.append("SupplyChainBridge")
except ImportError as e:
    SupplyChainBridge = None  # type: ignore[assignment,misc]
    logger.debug("SupplyChainBridge not available: %s", e)

# ---------------------------------------------------------------------------
# EUDR Bridge
# ---------------------------------------------------------------------------
try:
    from .eudr_bridge import (
        EUDRBridge,
        EUDRBridgeConfig,
        EUDRCommodity,
        DeforestationStatus,
        CountryBenchmark,
        DDSystemStatus,
        DeforestationAssessment,
        DeforestationStatusResult,
        CommodityRiskResult,
        RubberSourcingResult,
        COUNTRY_BENCHMARKS,
        BATTERY_RUBBER_COMPONENTS,
    )
    _loaded_integrations.append("EUDRBridge")
except ImportError as e:
    EUDRBridge = None  # type: ignore[assignment,misc]
    logger.debug("EUDRBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Taxonomy Bridge
# ---------------------------------------------------------------------------
try:
    from .taxonomy_bridge import (
        TaxonomyBridge,
        TaxonomyBridgeConfig,
        EnvironmentalObjective,
        DNSHStatus,
        AlignmentStatus,
        TaxonomyActivity,
        DNSHCriterion,
        DNSHResult,
        AlignmentResult,
        BatteryManufacturingCriteria,
        ACTIVITY_34_SC_CRITERIA,
        ACTIVITY_34_DNSH_CRITERIA,
        BATTERY_REG_CROSS_REFERENCES,
    )
    _loaded_integrations.append("TaxonomyBridge")
except ImportError as e:
    TaxonomyBridge = None  # type: ignore[assignment,misc]
    logger.debug("TaxonomyBridge not available: %s", e)

# ---------------------------------------------------------------------------
# CSDDD Bridge
# ---------------------------------------------------------------------------
try:
    from .csddd_bridge import (
        CSDDDBridge,
        CSDDDBridgeConfig,
        DDArticle,
        ImpactType,
        ImpactSeverity,
        DDComplianceLevel,
        BatteryMineral,
        AdverseImpact,
        DDStatusResult,
        AdverseImpactResult,
        MineralDDResult,
        CSDDD_BATTERY_OVERLAP,
        MINERAL_RISK_PROFILES,
    )
    _loaded_integrations.append("CSDDDBridge")
except ImportError as e:
    CSDDDBridge = None  # type: ignore[assignment,misc]
    logger.debug("CSDDDBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataBridge,
        DataBridgeConfig,
        DataSourceType,
        PassportDataCategory,
        QualityLevel,
        DataAgentMapping,
        IntakeResult,
        QualityReport,
        BATTERY_DATA_ROUTING,
        PASSPORT_FIELD_REQUIREMENTS,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    DataBridge = None  # type: ignore[assignment,misc]
    logger.debug("DataBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        BatteryPassportHealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        ComponentHealth,
        HealthSeverity,
        HealthStatus,
        CheckCategory,
        RemediationSuggestion,
    )
    _loaded_integrations.append("BatteryPassportHealthCheck")
except ImportError as e:
    BatteryPassportHealthCheck = None  # type: ignore[assignment,misc]
    logger.debug("BatteryPassportHealthCheck not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        BatteryPassportSetupWizard,
        SetupResult,
        SetupStatus,
        ManufacturerProfile,
        WizardConfig,
        BatteryCategory as WizardBatteryCategory,
        ChemistryType,
        RequirementsEstimate,
        CATEGORY_DEFAULTS,
    )
    _loaded_integrations.append("BatteryPassportSetupWizard")
except ImportError as e:
    BatteryPassportSetupWizard = None  # type: ignore[assignment,misc]
    logger.debug("BatteryPassportSetupWizard not available: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_loaded_integrations() -> List[str]:
    """Return list of successfully loaded integration class names."""
    return list(_loaded_integrations)


def get_integration_count() -> int:
    """Return count of loaded integrations."""
    return len(_loaded_integrations)


__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pipeline Orchestrator ---
    "BatteryPassportOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "BatteryPipelinePhase",
    "BatteryCategory",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    "QUICK_ASSESSMENT_PHASES",
    "REGULATION_ARTICLES",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVBridgeAgentMapping",
    "MRVScope",
    "LifecycleStage",
    "MRVAgentStatus",
    "ScopeEmissionsResult",
    "ManufacturingEmissionsResult",
    "CarbonIntensityResult",
    "BATTERY_MRV_ROUTING",
    "PERFORMANCE_CLASS_THRESHOLDS",
    # --- CSRD Pack Bridge ---
    "CSRDPackBridge",
    "CSRDBridgeConfig",
    "ESRSStandard",
    "ESRSDisclosureMapping",
    "ESRSImportResult",
    "MappingRelevance",
    "ImportStatus",
    "ESRS_BATTERY_MAPPINGS",
    # --- Supply Chain Bridge ---
    "SupplyChainBridge",
    "SupplyChainBridgeConfig",
    "SupplierTier",
    "CriticalMineral",
    "RiskLevel",
    "DDComplianceStatus",
    "SupplierRecord",
    "SupplierDataResult",
    "MineralSupplyChainResult",
    "SupplierRiskResult",
    "TierBreakdownResult",
    "CAHRA_COUNTRIES",
    # --- EUDR Bridge ---
    "EUDRBridge",
    "EUDRBridgeConfig",
    "EUDRCommodity",
    "DeforestationStatus",
    "CountryBenchmark",
    "DDSystemStatus",
    "DeforestationAssessment",
    "DeforestationStatusResult",
    "CommodityRiskResult",
    "RubberSourcingResult",
    "COUNTRY_BENCHMARKS",
    "BATTERY_RUBBER_COMPONENTS",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "EnvironmentalObjective",
    "DNSHStatus",
    "AlignmentStatus",
    "TaxonomyActivity",
    "DNSHCriterion",
    "DNSHResult",
    "AlignmentResult",
    "BatteryManufacturingCriteria",
    "ACTIVITY_34_SC_CRITERIA",
    "ACTIVITY_34_DNSH_CRITERIA",
    "BATTERY_REG_CROSS_REFERENCES",
    # --- CSDDD Bridge ---
    "CSDDDBridge",
    "CSDDDBridgeConfig",
    "DDArticle",
    "ImpactType",
    "ImpactSeverity",
    "DDComplianceLevel",
    "BatteryMineral",
    "AdverseImpact",
    "DDStatusResult",
    "AdverseImpactResult",
    "MineralDDResult",
    "CSDDD_BATTERY_OVERLAP",
    "MINERAL_RISK_PROFILES",
    # --- Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DataSourceType",
    "PassportDataCategory",
    "QualityLevel",
    "DataAgentMapping",
    "IntakeResult",
    "QualityReport",
    "BATTERY_DATA_ROUTING",
    "PASSPORT_FIELD_REQUIREMENTS",
    # --- Health Check ---
    "BatteryPassportHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Setup Wizard ---
    "BatteryPassportSetupWizard",
    "SetupResult",
    "SetupStatus",
    "ManufacturerProfile",
    "WizardConfig",
    "WizardBatteryCategory",
    "ChemistryType",
    "RequirementsEstimate",
    "CATEGORY_DEFAULTS",
    # --- Utility ---
    "get_loaded_integrations",
    "get_integration_count",
]
