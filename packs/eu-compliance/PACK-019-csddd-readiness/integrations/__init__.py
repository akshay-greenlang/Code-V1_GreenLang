# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Integration Layer
=====================================================

Integration bridges connecting the CSDDD Readiness Pack to other GreenLang
platform components including CSRD/ESRS packs, MRV agents, EUDR agents,
supply chain agents, data agents, EU Taxonomy, and Green Claims Directive.

Components:
    - CSDDDOrchestrator       : Master 7-phase assessment pipeline
    - CSRDPackBridge           : ESRS S1-S4/G1 to CSDDD mapping
    - MRVBridge                : MRV emission data for climate transition
    - EUDRBridge               : EUDR deforestation impact integration
    - SupplyChainBridge        : Value chain due diligence
    - DataBridge               : AGENT-DATA intake routing
    - GreenClaimsBridge        : Green Claims cross-validation
    - TaxonomyBridge           : EU Taxonomy DNSH alignment
    - CSDDDHealthCheck         : System health verification
    - CSDDDSetupWizard         : Guided configuration setup

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-019"
__pack_name__: str = "CSDDD Readiness Pack"

_loaded_integrations: List[str] = []

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        CSDDDOrchestrator,
        CSDDDPhase,
        ExecutionStatus,
        OrchestratorConfig,
        PhaseProvenance,
        PhaseResult,
        CompanyProfile as OrchestratorCompanyProfile,
        AssessmentResult,
        AssessmentSummary,
    )
    _loaded_integrations.append("CSDDDOrchestrator")
except ImportError as e:
    CSDDDOrchestrator = None  # type: ignore[assignment,misc]
    logger.debug("CSDDDOrchestrator not available: %s", e)

# ---------------------------------------------------------------------------
# CSRD Pack Bridge
# ---------------------------------------------------------------------------
try:
    from .csrd_pack_bridge import (
        CSRDPackBridge,
        ESRSStandard,
        BridgeConfig as CSRDBridgeConfig,
        ESRSDisclosure,
        CSDDDMapping,
        DisclosureGap,
        BridgeResult as CSRDBridgeResult,
        MappingCoverage,
        GapSeverity,
    )
    _loaded_integrations.append("CSRDPackBridge")
except ImportError as e:
    CSRDPackBridge = None  # type: ignore[assignment,misc]
    logger.debug("CSRDPackBridge not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVBridge,
        MRVBridgeConfig,
        EmissionScope,
        EmissionDataPoint,
        ScopeEmissions,
        TargetValidation,
        ReductionProgress,
        BridgeResult as MRVBridgeResult,
        TargetValidationStatus,
        ReductionTrajectory,
    )
    _loaded_integrations.append("MRVBridge")
except ImportError as e:
    MRVBridge = None  # type: ignore[assignment,misc]
    logger.debug("MRVBridge not available: %s", e)

# ---------------------------------------------------------------------------
# EUDR Bridge
# ---------------------------------------------------------------------------
try:
    from .eudr_bridge import (
        EUDRBridge,
        EUDRBridgeConfig,
        EUDRCommodity,
        DeforestationRisk,
        EUDRComplianceStatus,
        CSDDDImpactCategory,
        CommodityRiskProfile,
        DeforestationImpact,
        EUDRDueDiligenceStatus,
        CSDDDMappingResult,
    )
    _loaded_integrations.append("EUDRBridge")
except ImportError as e:
    EUDRBridge = None  # type: ignore[assignment,misc]
    logger.debug("EUDRBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Supply Chain Bridge
# ---------------------------------------------------------------------------
try:
    from .supply_chain_bridge import (
        SupplyChainBridge,
        SupplyChainBridgeConfig,
        SupplierTier,
        SupplierRiskLevel,
        RiskCategory,
        ValueChainDirection,
        SupplierProfile,
        ValueChainMap,
        SupplierRiskAssessment,
        BridgeResult as SupplyChainBridgeResult,
    )
    _loaded_integrations.append("SupplyChainBridge")
except ImportError as e:
    SupplyChainBridge = None  # type: ignore[assignment,misc]
    logger.debug("SupplyChainBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataBridge,
        DataBridgeConfig,
        DataSourceType,
        QualityLevel,
        FreshnessStatus,
        DataAgentId,
        DataSourceRecord,
        QuestionnaireData,
        SpendDataSummary,
        DataAggregation,
        DataQualityReport,
        BridgeResult as DataBridgeResult,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    DataBridge = None  # type: ignore[assignment,misc]
    logger.debug("DataBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Green Claims Bridge
# ---------------------------------------------------------------------------
try:
    from .green_claims_bridge import (
        GreenClaimsBridge,
        GreenClaimsBridgeConfig,
        ClaimType,
        SubstantiationStatus,
        GreenwashingRiskLevel,
        GreenwashingCategory,
        EnvironmentalClaim,
        ClaimValidationResult,
        GreenwashingAssessment,
        BridgeResult as GreenClaimsBridgeResult,
        CSDDDGreenClaimsMapping,
    )
    _loaded_integrations.append("GreenClaimsBridge")
except ImportError as e:
    GreenClaimsBridge = None  # type: ignore[assignment,misc]
    logger.debug("GreenClaimsBridge not available: %s", e)

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
        CSDDDEnvironmentalImpact,
        EconomicActivity,
        DNSHAssessment,
        EnvironmentalImpactMapping,
        TaxonomyAlignmentResult,
    )
    _loaded_integrations.append("TaxonomyBridge")
except ImportError as e:
    TaxonomyBridge = None  # type: ignore[assignment,misc]
    logger.debug("TaxonomyBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        CSDDDHealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        ComponentHealth,
        RemediationSuggestion,
        HealthSeverity,
        HealthStatus,
        CheckCategory,
    )
    _loaded_integrations.append("CSDDDHealthCheck")
except ImportError as e:
    CSDDDHealthCheck = None  # type: ignore[assignment,misc]
    logger.debug("CSDDDHealthCheck not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        CSDDDSetupWizard,
        WizardConfig,
        SetupResult,
        SetupStatus,
        SectorType,
        CompanyGroup,
        ValueChainDepth,
        CompanyProfile as WizardCompanyProfile,
        CSDDDConfiguration,
        ImplementationPhase,
        ImplementationPlan,
    )
    _loaded_integrations.append("CSDDDSetupWizard")
except ImportError as e:
    CSDDDSetupWizard = None  # type: ignore[assignment,misc]
    logger.debug("CSDDDSetupWizard not available: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_loaded_integrations() -> List[str]:
    """Return list of successfully loaded integration class names."""
    return list(_loaded_integrations)


def get_integration_count() -> int:
    """Return count of loaded integrations."""
    return len(_loaded_integrations)


__all__: List[str] = [
    # Module metadata
    "__version__",
    "__pack__",
    "__pack_name__",
    # --- Pipeline Orchestrator ---
    "CSDDDOrchestrator",
    "CSDDDPhase",
    "ExecutionStatus",
    "OrchestratorConfig",
    "PhaseProvenance",
    "PhaseResult",
    "OrchestratorCompanyProfile",
    "AssessmentResult",
    "AssessmentSummary",
    # --- CSRD Pack Bridge ---
    "CSRDPackBridge",
    "ESRSStandard",
    "CSRDBridgeConfig",
    "ESRSDisclosure",
    "CSDDDMapping",
    "DisclosureGap",
    "CSRDBridgeResult",
    "MappingCoverage",
    "GapSeverity",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "EmissionScope",
    "EmissionDataPoint",
    "ScopeEmissions",
    "TargetValidation",
    "ReductionProgress",
    "MRVBridgeResult",
    "TargetValidationStatus",
    "ReductionTrajectory",
    # --- EUDR Bridge ---
    "EUDRBridge",
    "EUDRBridgeConfig",
    "EUDRCommodity",
    "DeforestationRisk",
    "EUDRComplianceStatus",
    "CSDDDImpactCategory",
    "CommodityRiskProfile",
    "DeforestationImpact",
    "EUDRDueDiligenceStatus",
    "CSDDDMappingResult",
    # --- Supply Chain Bridge ---
    "SupplyChainBridge",
    "SupplyChainBridgeConfig",
    "SupplierTier",
    "SupplierRiskLevel",
    "RiskCategory",
    "ValueChainDirection",
    "SupplierProfile",
    "ValueChainMap",
    "SupplierRiskAssessment",
    "SupplyChainBridgeResult",
    # --- Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DataSourceType",
    "QualityLevel",
    "FreshnessStatus",
    "DataAgentId",
    "DataSourceRecord",
    "QuestionnaireData",
    "SpendDataSummary",
    "DataAggregation",
    "DataQualityReport",
    "DataBridgeResult",
    # --- Green Claims Bridge ---
    "GreenClaimsBridge",
    "GreenClaimsBridgeConfig",
    "ClaimType",
    "SubstantiationStatus",
    "GreenwashingRiskLevel",
    "GreenwashingCategory",
    "EnvironmentalClaim",
    "ClaimValidationResult",
    "GreenwashingAssessment",
    "GreenClaimsBridgeResult",
    "CSDDDGreenClaimsMapping",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "EnvironmentalObjective",
    "DNSHStatus",
    "AlignmentStatus",
    "CSDDDEnvironmentalImpact",
    "EconomicActivity",
    "DNSHAssessment",
    "EnvironmentalImpactMapping",
    "TaxonomyAlignmentResult",
    # --- Health Check ---
    "CSDDDHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "RemediationSuggestion",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    # --- Setup Wizard ---
    "CSDDDSetupWizard",
    "WizardConfig",
    "SetupResult",
    "SetupStatus",
    "SectorType",
    "CompanyGroup",
    "ValueChainDepth",
    "WizardCompanyProfile",
    "CSDDDConfiguration",
    "ImplementationPhase",
    "ImplementationPlan",
    # --- Utility ---
    "get_loaded_integrations",
    "get_integration_count",
]

logger.info(
    "PACK-019 integrations: %d/10 loaded",
    len(_loaded_integrations),
)
