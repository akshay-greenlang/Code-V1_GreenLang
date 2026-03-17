# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail & Consumer Goods Pack - Integration Layer
================================================================

Phase 4 integration layer for the CSRD Retail Pack that provides retail-
specific pipeline orchestration, cross-pack ESRS bridging, MRV agent
routing, ERP data integration, EUDR commodity tracing, circular economy
EPR compliance, supply chain due diligence, EU Taxonomy alignment,
22-category health verification, and 8-step retail setup wizard.

Components:
    - RetailPipelineOrchestrator: 11-phase retail pipeline with DAG
      dependency resolution, sub-sector phase skipping, retry with
      exponential backoff, and SHA-256 provenance tracking
    - CSRDPackBridge: Bridge to PACK-001/002/003 CSRD base packs for
      ESRS chapter assembly (E1, E5, S2, S4)
    - MRVRetailBridge: Routes retail emission sources to 30 MRV agents
      with sub-sector priority weighting
    - DataRetailBridge: Routes data intake to DATA agents and provides
      ERP field mapping for SAP/Oracle/NetSuite/Dynamics 365
    - EUDRRetailBridge: Maps retail products to EUDR commodities with
      multi-commodity tracing and country risk assessment
    - CircularEconomyBridge: EPR compliance, take-back tracking, and
      waste emissions calculation for WEEE/packaging/textile/battery
    - SupplyChainBridge: CSDDD due diligence, forced labour screening,
      and supplier remediation tracking
    - TaxonomyBridge: EU Taxonomy alignment for retail NACE activities
      with SC/DNSH criteria evaluation
    - RetailHealthCheck: 22-category system health verification
    - RetailSetupWizard: 8-step retail configuration with 13 sub-sector
      presets

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-014 Retail Pack <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- EUDR Agents <-- CSRD Packs

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/agents/data/* (20 DATA agents)
    - greenlang/agents/eudr/* (40 EUDR agents)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/eu-compliance/PACK-001 through PACK-003 (CSRD packs)
    - greenlang/apps/taxonomy (EU Taxonomy APP)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-014"
__pack_name__ = "CSRD Retail & Consumer Goods Pack"

# ---------------------------------------------------------------------------
# Retail Pipeline Orchestrator
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.pack_orchestrator import (
    ExecutionStatus,
    OrchestratorConfig,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetailPipelineOrchestrator,
    RetailPipelinePhase,
    RetailSubSector,
    RetryConfig,
)

# ---------------------------------------------------------------------------
# CSRD Pack Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.csrd_pack_bridge import (
    BasePack,
    BridgeResult,
    CSRDPackBridge,
    CSRDPackBridgeConfig,
    DatapointMapping,
    ESRSChapter,
    ESRSChapterData,
)

# ---------------------------------------------------------------------------
# MRV Retail Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.mrv_retail_bridge import (
    BatchRoutingResult,
    EmissionSource,
    MRVAgentRoute,
    MRVBridgeConfig,
    MRVRetailBridge,
    MRVScope,
    RoutingResult,
)

# ---------------------------------------------------------------------------
# Data Retail Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.data_retail_bridge import (
    DataAgentRoute,
    DataBridgeConfig,
    DataRetailBridge,
    DataRoutingResult,
    DataSource,
    ERPExtractionResult,
    ERPFieldMapping,
    RetailERP,
)

# ---------------------------------------------------------------------------
# EUDR Retail Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.eudr_retail_bridge import (
    EUDRBridgeConfig,
    EUDRCommodity,
    EUDRRetailBridge,
    MultiCommodityResult,
    ProductCommodityMapping,
    RiskLevel,
    TraceabilityResult,
)

# ---------------------------------------------------------------------------
# Circular Economy Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.circular_economy_bridge import (
    CircularBridgeConfig,
    CircularEconomyBridge,
    EPRComplianceResult,
    EPRScheme,
    TakeBackResult,
    WasteEmissionsResult,
    WasteStream,
)

# ---------------------------------------------------------------------------
# Supply Chain Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.supply_chain_bridge import (
    DueDiligenceResult,
    RemediationAction,
    RemediationStatus,
    RiskCategory,
    SupplierRiskResult,
    SupplierTier,
    SupplyChainBridge,
    SupplyChainBridgeConfig,
)

# ---------------------------------------------------------------------------
# Taxonomy Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.taxonomy_bridge import (
    AlignmentLevel,
    CriterionStatus,
    DNSHCriterion,
    RetailNACEActivity,
    SCCriterion,
    TaxonomyAssessmentResult,
    TaxonomyBridge,
    TaxonomyBridgeConfig,
    TaxonomyObjective,
)

# ---------------------------------------------------------------------------
# Retail Health Check
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.health_check import (
    CheckCategory,
    ComponentHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    RemediationSuggestion,
    RetailHealthCheck,
)

# ---------------------------------------------------------------------------
# Retail Setup Wizard
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_014_csrd_retail.integrations.setup_wizard import (
    CompanyProfile,
    EmissionsSourceConfig,
    ProductCategoryConfig,
    RegulatoryScope,
    ReportingSetup,
    RetailSetupWizard,
    RetailSubSectorConfig,
    RetailWizardStep,
    SetupResult,
    StepStatus,
    StorePortfolio,
    SupplyChainConfig,
    WizardState,
    WizardStepState,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Retail Pipeline Orchestrator ---
    "RetailPipelineOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "RetailPipelinePhase",
    "RetailSubSector",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    # --- CSRD Pack Bridge ---
    "CSRDPackBridge",
    "CSRDPackBridgeConfig",
    "DatapointMapping",
    "ESRSChapterData",
    "ESRSChapter",
    "BasePack",
    "BridgeResult",
    # --- MRV Retail Bridge ---
    "MRVRetailBridge",
    "MRVBridgeConfig",
    "MRVAgentRoute",
    "EmissionSource",
    "MRVScope",
    "RoutingResult",
    "BatchRoutingResult",
    # --- Data Retail Bridge ---
    "DataRetailBridge",
    "DataBridgeConfig",
    "DataAgentRoute",
    "DataSource",
    "RetailERP",
    "ERPFieldMapping",
    "DataRoutingResult",
    "ERPExtractionResult",
    # --- EUDR Retail Bridge ---
    "EUDRRetailBridge",
    "EUDRBridgeConfig",
    "EUDRCommodity",
    "RiskLevel",
    "ProductCommodityMapping",
    "TraceabilityResult",
    "MultiCommodityResult",
    # --- Circular Economy Bridge ---
    "CircularEconomyBridge",
    "CircularBridgeConfig",
    "EPRScheme",
    "WasteStream",
    "EPRComplianceResult",
    "TakeBackResult",
    "WasteEmissionsResult",
    # --- Supply Chain Bridge ---
    "SupplyChainBridge",
    "SupplyChainBridgeConfig",
    "RiskCategory",
    "SupplierTier",
    "RemediationStatus",
    "SupplierRiskResult",
    "RemediationAction",
    "DueDiligenceResult",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyObjective",
    "AlignmentLevel",
    "CriterionStatus",
    "RetailNACEActivity",
    "SCCriterion",
    "DNSHCriterion",
    "TaxonomyAssessmentResult",
    # --- Retail Health Check ---
    "RetailHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Retail Setup Wizard ---
    "RetailSetupWizard",
    "RetailWizardStep",
    "StepStatus",
    "CompanyProfile",
    "StorePortfolio",
    "RetailSubSectorConfig",
    "RegulatoryScope",
    "EmissionsSourceConfig",
    "ProductCategoryConfig",
    "SupplyChainConfig",
    "ReportingSetup",
    "WizardStepState",
    "WizardState",
    "SetupResult",
]
