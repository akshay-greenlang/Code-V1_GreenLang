# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Integration Layer
========================================================

Phase 4 integration layer for the ESRS Full Coverage Pack that provides
15-phase pipeline orchestration across all 12 ESRS standards, PACK-016
E1 Climate bridging, PACK-015 DMA materiality bridging, GL-CSRD-APP
datapoint/formula/rule import, MRV agent emissions routing (30 agents),
DATA agent intake/quality routing (20 agents), EU Taxonomy alignment,
EFRAG XBRL/iXBRL taxonomy tagging, 20-category health verification,
and interactive guided setup wizard.

Components:
    - ESRSFullOrchestrator: 15-phase pipeline with DAG dependency
      resolution, parallel standard execution, materiality gating,
      retry with exponential backoff, and SHA-256 provenance tracking
    - E1PackBridge: Bridge to PACK-016 for E1 Climate data import,
      CDM mapping, compliance scoring, and XBRL datapoints
    - DMAPackBridge: Bridge to PACK-015 for materiality import,
      standard activation gating, and IRO register
    - CSRDAppBridge: Bridge to GL-CSRD-APP for 1,093 datapoints,
      524 formulas, 235 validation rules, and XBRL taxonomy
    - MRVAgentBridge: Routes emissions from 30 MRV agents for
      Scope 1/2/3 data plus E2 pollution and E5 waste data
    - DataAgentBridge: Routes data intake through 20 DATA agents
      with ERP field mappings for SAP/Oracle/Workday/MS Dynamics
    - TaxonomyBridge: EU Taxonomy alignment with 6 environmental
      objectives, CapEx/OpEx/Revenue KPIs, and TSC mapping
    - XBRLTaggingBridge: EFRAG XBRL taxonomy mapping with 11
      namespaces and iXBRL inline tag generation
    - ESRSHealthCheck: 20-category system health verification
    - PackSetupWizard: Interactive configuration setup wizard

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-017 Full Coverage <-- Composition <-- Zero Duplication
                              |
                              v
    PACK-016 (E1) <-- PACK-015 (DMA) <-- GL-CSRD-APP
    MRV Agents (30) <-- DATA Agents (20) <-- FOUND Agents (10)
    EU Taxonomy <-- XBRL Taxonomy

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/agents/data/* (20 DATA agents)
    - greenlang/apps/GL-CSRD-APP (CSRD application)
    - packs/eu-compliance/PACK-015 (Double Materiality)
    - packs/eu-compliance/PACK-016 (ESRS E1 Climate)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-017 ESRS Full Coverage Pack
Status: Production Ready
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__pack_id__ = "PACK-017"
__pack_name__ = "ESRS Full Coverage Pack"

_loaded_integrations: List[str] = []

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        ESRSFullOrchestrator,
        ESRSPipelinePhase,
        ExecutionStatus,
        OrchestratorConfig,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PARALLEL_PHASE_GROUPS,
        PhaseProvenance,
        PhaseResult,
        PipelineResult,
        RetryConfig,
    )
    _loaded_integrations.append("ESRSFullOrchestrator")
except ImportError as e:
    ESRSFullOrchestrator = None  # type: ignore[assignment,misc]
    logger.debug("ESRSFullOrchestrator not available: %s", e)

# ---------------------------------------------------------------------------
# E1 Pack Bridge (PACK-016)
# ---------------------------------------------------------------------------
try:
    from .e1_pack_bridge import (
        E1PackBridge,
        E1BridgeConfig,
        E1ComplianceScore,
        E1DataPoint,
        E1DisclosureRequirement,
        E1ImportResult,
        E1ImportStatus,
    )
    _loaded_integrations.append("E1PackBridge")
except ImportError as e:
    E1PackBridge = None  # type: ignore[assignment,misc]
    logger.debug("E1PackBridge not available: %s", e)

# ---------------------------------------------------------------------------
# DMA Pack Bridge (PACK-015)
# ---------------------------------------------------------------------------
try:
    from .dma_pack_bridge import (
        DMAPackBridge,
        DMABridgeConfig,
        DMAImportResult,
        ESRSStandard,
        IROEntry,
        IROType,
        MaterialityStatus,
        StandardMateriality,
    )
    _loaded_integrations.append("DMAPackBridge")
except ImportError as e:
    DMAPackBridge = None  # type: ignore[assignment,misc]
    logger.debug("DMAPackBridge not available: %s", e)

# ---------------------------------------------------------------------------
# CSRD App Bridge
# ---------------------------------------------------------------------------
try:
    from .csrd_app_bridge import (
        CSRDAppBridge,
        CSRDBridgeConfig,
        DataPointType,
        DisclosureStatus,
        ESRSDataPoint,
        ESRSFormula,
        MandatoryLevel,
        ValidationRule,
        XBRLTaxonomyElement,
    )
    _loaded_integrations.append("CSRDAppBridge")
except ImportError as e:
    CSRDAppBridge = None  # type: ignore[assignment,misc]
    logger.debug("CSRDAppBridge not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Agent Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_agent_bridge import (
        MRVAgentBridge,
        MRVBridgeConfig,
        MRVAgentMapping,
        MRVScope,
        AgentStatus as MRVAgentStatus,
        AggregationResult,
        ScopeImportResult,
        SCOPE1_AGENTS,
        SCOPE2_AGENTS,
        SCOPE3_AGENTS,
        CROSS_CUTTING_AGENTS,
        MRV_AGENT_ROUTING,
    )
    _loaded_integrations.append("MRVAgentBridge")
except ImportError as e:
    MRVAgentBridge = None  # type: ignore[assignment,misc]
    logger.debug("MRVAgentBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Data Agent Bridge
# ---------------------------------------------------------------------------
try:
    from .data_agent_bridge import (
        DataAgentBridge,
        DataBridgeConfig,
        DataAgentMapping,
        DataSourceType,
        ERPSystem,
        IntakeResult,
        QualityLevel,
        QualityReport,
        DATA_AGENT_ROUTING,
        ERP_FIELD_MAPPINGS,
    )
    _loaded_integrations.append("DataAgentBridge")
except ImportError as e:
    DataAgentBridge = None  # type: ignore[assignment,misc]
    logger.debug("DataAgentBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Taxonomy Bridge
# ---------------------------------------------------------------------------
try:
    from .taxonomy_bridge import (
        TaxonomyBridge,
        TaxonomyBridgeConfig,
        AlignmentKPIs,
        AlignmentStatus,
        DNSHStatus,
        EnvironmentalObjective,
        TaxonomyActivity,
        ESRS_TO_TAXONOMY_MAP,
        TSC_REFERENCE_MAP,
    )
    _loaded_integrations.append("TaxonomyBridge")
except ImportError as e:
    TaxonomyBridge = None  # type: ignore[assignment,misc]
    logger.debug("TaxonomyBridge not available: %s", e)

# ---------------------------------------------------------------------------
# XBRL Tagging Bridge
# ---------------------------------------------------------------------------
try:
    from .xbrl_tagging_bridge import (
        XBRLTaggingBridge,
        XBRLBridgeConfig,
        XBRLDataType,
        XBRLElement,
        XBRLTag,
        TaggingResult,
        ValidationResult as XBRLValidationResult,
        XBRL_NAMESPACE_MAP,
        STANDARD_NAMESPACE_MAP,
    )
    _loaded_integrations.append("XBRLTaggingBridge")
except ImportError as e:
    XBRLTaggingBridge = None  # type: ignore[assignment,misc]
    logger.debug("XBRLTaggingBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        ESRSHealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        ComponentHealth,
        HealthSeverity,
        HealthStatus,
        CheckCategory,
        RemediationSuggestion,
    )
    _loaded_integrations.append("ESRSHealthCheck")
except ImportError as e:
    ESRSHealthCheck = None  # type: ignore[assignment,misc]
    logger.debug("ESRSHealthCheck not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        PackSetupWizard,
        SetupResult,
        SetupStatus,
        OrganizationProfile,
        WizardConfig,
        SectorType,
        MaterialityLevel,
        MaterialityPreferences,
    )
    _loaded_integrations.append("PackSetupWizard")
except ImportError as e:
    PackSetupWizard = None  # type: ignore[assignment,misc]
    logger.debug("PackSetupWizard not available: %s", e)


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
    "ESRSFullOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "ESRSPipelinePhase",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- E1 Pack Bridge ---
    "E1PackBridge",
    "E1BridgeConfig",
    "E1ComplianceScore",
    "E1DataPoint",
    "E1DisclosureRequirement",
    "E1ImportResult",
    "E1ImportStatus",
    # --- DMA Pack Bridge ---
    "DMAPackBridge",
    "DMABridgeConfig",
    "DMAImportResult",
    "ESRSStandard",
    "IROEntry",
    "IROType",
    "MaterialityStatus",
    "StandardMateriality",
    # --- CSRD App Bridge ---
    "CSRDAppBridge",
    "CSRDBridgeConfig",
    "DataPointType",
    "DisclosureStatus",
    "ESRSDataPoint",
    "ESRSFormula",
    "MandatoryLevel",
    "ValidationRule",
    "XBRLTaxonomyElement",
    # --- MRV Agent Bridge ---
    "MRVAgentBridge",
    "MRVBridgeConfig",
    "MRVAgentMapping",
    "MRVScope",
    "MRVAgentStatus",
    "AggregationResult",
    "ScopeImportResult",
    "SCOPE1_AGENTS",
    "SCOPE2_AGENTS",
    "SCOPE3_AGENTS",
    "CROSS_CUTTING_AGENTS",
    "MRV_AGENT_ROUTING",
    # --- Data Agent Bridge ---
    "DataAgentBridge",
    "DataBridgeConfig",
    "DataAgentMapping",
    "DataSourceType",
    "ERPSystem",
    "IntakeResult",
    "QualityLevel",
    "QualityReport",
    "DATA_AGENT_ROUTING",
    "ERP_FIELD_MAPPINGS",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "AlignmentKPIs",
    "AlignmentStatus",
    "DNSHStatus",
    "EnvironmentalObjective",
    "TaxonomyActivity",
    "ESRS_TO_TAXONOMY_MAP",
    "TSC_REFERENCE_MAP",
    # --- XBRL Tagging Bridge ---
    "XBRLTaggingBridge",
    "XBRLBridgeConfig",
    "XBRLDataType",
    "XBRLElement",
    "XBRLTag",
    "TaggingResult",
    "XBRLValidationResult",
    "XBRL_NAMESPACE_MAP",
    "STANDARD_NAMESPACE_MAP",
    # --- Health Check ---
    "ESRSHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Setup Wizard ---
    "PackSetupWizard",
    "SetupResult",
    "SetupStatus",
    "OrganizationProfile",
    "WizardConfig",
    "SectorType",
    "MaterialityLevel",
    "MaterialityPreferences",
    # --- Utility ---
    "get_loaded_integrations",
    "get_integration_count",
]
