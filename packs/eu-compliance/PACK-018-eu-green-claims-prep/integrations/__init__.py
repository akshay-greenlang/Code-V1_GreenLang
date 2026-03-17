# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Integration Layer
=========================================================

Phase 4 integration layer for the EU Green Claims Prep Pack that provides
10-phase pipeline orchestration for environmental marketing claim
verification, CSRD data bridging (PACK-001/002/003), MRV agent emissions
routing (30 agents), DATA agent evidence gathering (20 agents), EU
Taxonomy alignment, Product Environmental Footprint (PEF) data exchange,
Digital Product Passport (DPP) integration, ECGT Directive compliance
checking, 20-category health verification, and interactive setup wizard.

Components:
    - GreenClaimsOrchestrator: 10-phase pipeline with DAG dependency
      resolution, phase enable/disable flags, and SHA-256 provenance
    - CSRDPackBridge: Bridge to PACK-001/002/003 for CSRD evidence
    - MRVClaimsBridge: Routes carbon claims to 30 MRV agents
    - DataClaimsBridge: Routes evidence requests to 20 DATA agents
    - TaxonomyBridge: EU Taxonomy alignment for "sustainable" claims
    - PEFBridge: Product Environmental Footprint data exchange
    - DPPBridge: Digital Product Passport integration per ESPR
    - ECGTBridge: ECGT Directive prohibited practices checking
    - GreenClaimsHealthCheck: 20-category system health verification
    - GreenClaimsSetupWizard: 8-step configuration wizard

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-018 Green Claims <-- Composition <-- Zero Duplication
                              |
                              v
    PACK-001/002/003 (CSRD) <-- MRV Agents (30) <-- DATA Agents (20)
    EU Taxonomy <-- PEF Studies <-- DPP Registry <-- ECGT Directive

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/agents/data/* (20 DATA agents)
    - packs/eu-compliance/PACK-001 (CSRD Starter)
    - packs/eu-compliance/PACK-002 (CSRD Professional)
    - packs/eu-compliance/PACK-003 (CSRD Enterprise)
    - packs/eu-compliance/PACK-008 (EU Taxonomy Alignment)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__pack_id__ = "PACK-018"
__pack_name__ = "EU Green Claims Prep Pack"

_loaded_integrations: List[str] = []

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        GreenClaimsOrchestrator,
        ClaimsPipelinePhase,
        ExecutionStatus,
        OrchestratorConfig,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PARALLEL_PHASE_GROUPS,
        PhaseResult,
        PipelineResult,
    )
    _loaded_integrations.append("GreenClaimsOrchestrator")
except ImportError as e:
    GreenClaimsOrchestrator = None  # type: ignore[assignment,misc]
    logger.debug("GreenClaimsOrchestrator not available: %s", e)

# ---------------------------------------------------------------------------
# CSRD Pack Bridge
# ---------------------------------------------------------------------------
try:
    from .csrd_pack_bridge import (
        CSRDPackBridge,
        CSRDBridgeConfig,
        CSRDBridgeResult,
        CSRDEvidenceMapping,
        CSRDPackTier,
        ESRSDataCategory,
        BridgeStatus,
        ESRS_CLAIMS_MAP,
    )
    _loaded_integrations.append("CSRDPackBridge")
except ImportError as e:
    CSRDPackBridge = None  # type: ignore[assignment,misc]
    logger.debug("CSRDPackBridge not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Claims Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_claims_bridge import (
        MRVClaimsBridge,
        MRVRoutingConfig,
        MRVRoutingResult,
        AgentRoutingEntry,
        GHGScope,
        ClaimVerificationStatus,
        RoutingStatus as MRVRoutingStatus,
        SCOPE1_AGENTS,
        SCOPE2_AGENTS,
        SCOPE3_AGENTS,
        CROSS_CUTTING_AGENTS,
        CLAIM_TO_AGENT_MAP,
    )
    _loaded_integrations.append("MRVClaimsBridge")
except ImportError as e:
    MRVClaimsBridge = None  # type: ignore[assignment,misc]
    logger.debug("MRVClaimsBridge not available: %s", e)

# ---------------------------------------------------------------------------
# DATA Claims Bridge
# ---------------------------------------------------------------------------
try:
    from .data_claims_bridge import (
        DataClaimsBridge,
        DataRoutingConfig,
        DataRoutingResult,
        DataRoutingEntry,
        EvidenceSourceType,
        DataQualityLevel,
        RoutingStatus as DataRoutingStatus,
        INTAKE_AGENTS,
        QUALITY_AGENTS,
        GEO_AGENTS,
        EVIDENCE_TO_AGENT_MAP,
    )
    _loaded_integrations.append("DataClaimsBridge")
except ImportError as e:
    DataClaimsBridge = None  # type: ignore[assignment,misc]
    logger.debug("DataClaimsBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Taxonomy Bridge
# ---------------------------------------------------------------------------
try:
    from .taxonomy_bridge import (
        TaxonomyBridge,
        TaxonomyBridgeConfig,
        TaxonomyAlignmentResult,
        ObjectiveAssessment,
        AlignmentKPIs,
        AlignmentStatus,
        DNSHStatus,
        EnvironmentalObjective,
        MinimumSafeguardStatus,
        CLAIM_TO_OBJECTIVE_MAP,
        TSC_REFERENCE_MAP,
    )
    _loaded_integrations.append("TaxonomyBridge")
except ImportError as e:
    TaxonomyBridge = None  # type: ignore[assignment,misc]
    logger.debug("TaxonomyBridge not available: %s", e)

# ---------------------------------------------------------------------------
# PEF Bridge
# ---------------------------------------------------------------------------
try:
    from .pef_bridge import (
        PEFBridge,
        PEFBridgeConfig,
        PEFDataResult,
        ImpactResult,
        PEFImpactCategory,
        LifecycleStage,
        PEFStudyStatus,
        IMPACT_UNITS,
        PEFCR_REGISTRY,
    )
    _loaded_integrations.append("PEFBridge")
except ImportError as e:
    PEFBridge = None  # type: ignore[assignment,misc]
    logger.debug("PEFBridge not available: %s", e)

# ---------------------------------------------------------------------------
# DPP Bridge
# ---------------------------------------------------------------------------
try:
    from .dpp_bridge import (
        DPPBridge,
        DPPBridgeConfig,
        DPPLinkingResult,
        DPPDataSnapshot,
        ProductGroup,
        PassportSchemaVersion,
        LinkingStatus,
        DPPDataField,
        PRODUCT_GROUP_DPP_FIELDS,
        CLAIM_TO_DPP_FIELDS,
    )
    _loaded_integrations.append("DPPBridge")
except ImportError as e:
    DPPBridge = None  # type: ignore[assignment,misc]
    logger.debug("DPPBridge not available: %s", e)

# ---------------------------------------------------------------------------
# ECGT Bridge
# ---------------------------------------------------------------------------
try:
    from .ecgt_bridge import (
        ECGTBridge,
        ECGTBridgeConfig,
        ECGTComplianceResult,
        ProhibitedPracticeDetection,
        LabelCheckResult,
        ProhibitedPractice,
        ECGTCheckStatus,
        LabelVerificationStatus,
        GENERIC_CLAIM_KEYWORDS,
        KNOWN_EU_LABELS,
    )
    _loaded_integrations.append("ECGTBridge")
except ImportError as e:
    ECGTBridge = None  # type: ignore[assignment,misc]
    logger.debug("ECGTBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        GreenClaimsHealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        ComponentHealth,
        RemediationSuggestion,
        HealthSeverity,
        HealthStatus,
        HealthCheckCategory,
    )
    _loaded_integrations.append("GreenClaimsHealthCheck")
except ImportError as e:
    GreenClaimsHealthCheck = None  # type: ignore[assignment,misc]
    logger.debug("GreenClaimsHealthCheck not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        GreenClaimsSetupWizard,
        WizardConfig,
        WizardState,
        WizardStep,
        StepResult,
        StepStatus,
        SectorPreset,
    )
    _loaded_integrations.append("GreenClaimsSetupWizard")
except ImportError as e:
    GreenClaimsSetupWizard = None  # type: ignore[assignment,misc]
    logger.debug("GreenClaimsSetupWizard not available: %s", e)


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
    "GreenClaimsOrchestrator",
    "OrchestratorConfig",
    "ClaimsPipelinePhase",
    "ExecutionStatus",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- CSRD Pack Bridge ---
    "CSRDPackBridge",
    "CSRDBridgeConfig",
    "CSRDBridgeResult",
    "CSRDEvidenceMapping",
    "CSRDPackTier",
    "ESRSDataCategory",
    "BridgeStatus",
    "ESRS_CLAIMS_MAP",
    # --- MRV Claims Bridge ---
    "MRVClaimsBridge",
    "MRVRoutingConfig",
    "MRVRoutingResult",
    "AgentRoutingEntry",
    "GHGScope",
    "ClaimVerificationStatus",
    "MRVRoutingStatus",
    "SCOPE1_AGENTS",
    "SCOPE2_AGENTS",
    "SCOPE3_AGENTS",
    "CROSS_CUTTING_AGENTS",
    "CLAIM_TO_AGENT_MAP",
    # --- DATA Claims Bridge ---
    "DataClaimsBridge",
    "DataRoutingConfig",
    "DataRoutingResult",
    "DataRoutingEntry",
    "EvidenceSourceType",
    "DataQualityLevel",
    "DataRoutingStatus",
    "INTAKE_AGENTS",
    "QUALITY_AGENTS",
    "GEO_AGENTS",
    "EVIDENCE_TO_AGENT_MAP",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyAlignmentResult",
    "ObjectiveAssessment",
    "AlignmentKPIs",
    "AlignmentStatus",
    "DNSHStatus",
    "EnvironmentalObjective",
    "MinimumSafeguardStatus",
    "CLAIM_TO_OBJECTIVE_MAP",
    "TSC_REFERENCE_MAP",
    # --- PEF Bridge ---
    "PEFBridge",
    "PEFBridgeConfig",
    "PEFDataResult",
    "ImpactResult",
    "PEFImpactCategory",
    "LifecycleStage",
    "PEFStudyStatus",
    "IMPACT_UNITS",
    "PEFCR_REGISTRY",
    # --- DPP Bridge ---
    "DPPBridge",
    "DPPBridgeConfig",
    "DPPLinkingResult",
    "DPPDataSnapshot",
    "ProductGroup",
    "PassportSchemaVersion",
    "LinkingStatus",
    "DPPDataField",
    "PRODUCT_GROUP_DPP_FIELDS",
    "CLAIM_TO_DPP_FIELDS",
    # --- ECGT Bridge ---
    "ECGTBridge",
    "ECGTBridgeConfig",
    "ECGTComplianceResult",
    "ProhibitedPracticeDetection",
    "LabelCheckResult",
    "ProhibitedPractice",
    "ECGTCheckStatus",
    "LabelVerificationStatus",
    "GENERIC_CLAIM_KEYWORDS",
    "KNOWN_EU_LABELS",
    # --- Health Check ---
    "GreenClaimsHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "RemediationSuggestion",
    "HealthSeverity",
    "HealthStatus",
    "HealthCheckCategory",
    # --- Setup Wizard ---
    "GreenClaimsSetupWizard",
    "WizardConfig",
    "WizardState",
    "WizardStep",
    "StepResult",
    "StepStatus",
    "SectorPreset",
    # --- Utility ---
    "get_loaded_integrations",
    "get_integration_count",
]
