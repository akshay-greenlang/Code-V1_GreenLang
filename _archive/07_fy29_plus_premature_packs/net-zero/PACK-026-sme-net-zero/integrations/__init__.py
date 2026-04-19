# -*- coding: utf-8 -*-
"""
PACK-026 SME Net Zero Pack - Integration Layer
=================================================

Phase 4 integration layer for the SME Net Zero Pack providing
pipeline orchestration, simplified MRV agent routing, SME accounting
software connectors (Xero, QuickBooks, Sage), grant database search,
SME Climate Hub submission, multi-certification management, peer
benchmarking, renewable PPA marketplace, 5-step setup wizard, and
12-category health verification.

Components:
    - SMENetZeroPipelineOrchestrator: 6-phase DAG pipeline for SME
      workflows (Onboarding -> Baseline -> Targets -> Quick Wins ->
      Grant Search -> Reporting) with simplified/standard path selection
    - SMEMRVBridge: Routes to SME-relevant subset of 30 MRV agents
      (7 agents: stationary combustion, mobile, electricity, natural gas,
      business travel, employee commuting, spend-based Scope 3)
    - SMEDataBridge: DATA agent integration with spend categorization,
      auto-mapping to emission categories, and industry defaults
    - XeroConnector: Xero API integration (OAuth2) with chart of accounts,
      transaction export, GL code mapping, and rate limiting (5 req/sec)
    - QuickBooksConnector: QuickBooks Online API integration (OAuth2)
      with P&L export, spend categorization, and rate limiting
    - SageConnector: Sage Business Cloud API integration with nominal
      ledger export, multi-currency support, and rate limiting
    - GrantDatabaseBridge: UK/EU/US/AU/NZ/CA grant matching with
      eligibility scoring, deadline tracking, and application management
    - SMEClimateHubBridge: UN SME Climate Hub commitment submission,
      annual progress reporting, verification status, and badge management
    - CertificationBodyBridge: Multi-certification integration for B Corp,
      Carbon Trust, ISO 14001, and Climate Active (Australia)
    - PeerNetworkBridge: Anonymous peer benchmarking with aggregated
      industry statistics by sector, size tier, and geography
    - RenewablePPAMarketplace: PPA aggregator integration with contract
      search, cost comparison, and Scope 2 reduction calculation
    - SMESetupWizard: 5-step guided configuration (org profile, data
      quality tier, accounting connection, grant preferences, certification)
    - SMEHealthCheck: 12-category system health verification

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-026 SME Net Zero <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (subset) <-- DATA Agents (subset) <-- Accounting APIs
                              |
                              v
    Grant DBs <-- SME Climate Hub <-- Certification Bodies <-- PPA Market

Platform Integrations:
    - greenlang/agents/mrv/* (7 of 30 MRV agents)
    - greenlang/agents/data/* (6 of 20 DATA agents)
    - Xero API (xero.com)
    - QuickBooks Online API (intuit.com)
    - Sage Business Cloud API (sage.com)
    - SME Climate Hub API (smeclimatehub.org)
    - B Corp API (bcorporation.net)
    - Carbon Trust API (carbontrust.com)
    - Grant databases (BEIS, GFI, LIFE, DOE, EPA, etc.)
    - PPA aggregators (Arcano, Enel, NextEnergy)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-026"
__pack_name__ = "SME Net Zero Pack"

# ---------------------------------------------------------------------------
# SME Net Zero Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    DataQualityTier,
    ExecutionStatus,
    PhaseProgress,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
    SMENetZeroPipelineOrchestrator,
    SMEOrchestratorConfig,
    SMEPathType,
    SMEPipelinePhase,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
)

# ---------------------------------------------------------------------------
# SME MRV Bridge
# ---------------------------------------------------------------------------
from .mrv_bridge import (
    BatchRoutingResult,
    MRVScope,
    RoutingResult,
    SMEActivityData,
    SMEEmissionSource,
    SMEMRVAgentRoute,
    SMEMRVBridge,
    SMEMRVBridgeConfig,
    ValidationResult as MRVValidationResult,
)

# ---------------------------------------------------------------------------
# SME Data Bridge
# ---------------------------------------------------------------------------
from .data_bridge import (
    IntakeResult,
    QualityResult,
    SMEDataBridge,
    SMEDataBridgeConfig,
    SMEDataCategory,
    SMEDataSourceType,
    SMESpendCategory,
    SpendMappingResult,
)

# ---------------------------------------------------------------------------
# Xero Connector
# ---------------------------------------------------------------------------
from .xero_connector import (
    XeroAccount,
    XeroAggregation,
    XeroConfig,
    XeroConnectionStatus,
    XeroConnector,
    XeroExportResult,
    XeroTransaction,
)

# ---------------------------------------------------------------------------
# QuickBooks Connector
# ---------------------------------------------------------------------------
from .quickbooks_connector import (
    QBAccount,
    QBAggregation,
    QBConfig,
    QBConnectionStatus,
    QBExportResult,
    QBTransaction,
    QuickBooksConnector,
)

# ---------------------------------------------------------------------------
# Sage Connector
# ---------------------------------------------------------------------------
from .sage_connector import (
    SageAggregation,
    SageConfig,
    SageConnectionStatus,
    SageConnector,
    SageExportResult,
    SageNominalAccount,
    SageTransaction,
)

# ---------------------------------------------------------------------------
# Grant Database Bridge
# ---------------------------------------------------------------------------
from .grant_database_bridge import (
    ApplicationStatus,
    DeadlineAlert,
    GrantApplication,
    GrantCategory,
    GrantDatabaseBridge,
    GrantDatabaseConfig,
    GrantMatch,
    GrantRegion,
    GrantSearchResult,
    GrantStatus,
)

# ---------------------------------------------------------------------------
# SME Climate Hub Bridge
# ---------------------------------------------------------------------------
from .sme_climate_hub_bridge import (
    BadgeInfo,
    BadgeType,
    CommitmentData,
    CommitmentResult,
    CommitmentStatus,
    ProgressReport,
    ProgressStatus,
    ProgressSubmissionResult,
    SMEClimateHubBridge,
    SMEClimateHubConfig,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Certification Body Bridge
# ---------------------------------------------------------------------------
from .certification_body_bridge import (
    CertificationBodyBridge,
    CertificationBodyConfig,
    CertificationReadiness,
    CertificationStatus,
    CertificationSubmission,
    CertificationType,
    DocumentType,
    DocumentUploadResult,
)

# ---------------------------------------------------------------------------
# Peer Network Bridge
# ---------------------------------------------------------------------------
from .peer_network_bridge import (
    BenchmarkMetric,
    BenchmarkResult,
    PeerNetworkBridge,
    PeerNetworkConfig,
    PercentileRanking,
    SizeTier,
)

# ---------------------------------------------------------------------------
# Renewable PPA Marketplace
# ---------------------------------------------------------------------------
from .renewable_ppa_marketplace import (
    ComparisonTable,
    CostComparison,
    EnergySource,
    PPAContract,
    PPAInterest,
    PPAMarketplaceConfig,
    PPAProvider,
    PPASearchResult,
    PPAType,
    RenewablePPAMarketplace,
)

# ---------------------------------------------------------------------------
# SME Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    AccountingConnectionSetup,
    AccountingSoftware,
    CertificationPathway,
    CertificationSelection,
    DataQualitySelection,
    GrantPreferences,
    SMEOrganizationProfile,
    SMESetupWizard,
    SMESize,
    SMEWizardStep,
    SetupResult,
    StepStatus,
    WizardState,
    WizardStepState,
)

# ---------------------------------------------------------------------------
# SME Health Check
# ---------------------------------------------------------------------------
from .health_check import (
    CheckCategory,
    ComponentHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    RemediationSuggestion,
    SMEHealthCheck,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- SME Pipeline Orchestrator ---
    "SMENetZeroPipelineOrchestrator",
    "SMEOrchestratorConfig",
    "RetryConfig",
    "SMEPipelinePhase",
    "SMEPathType",
    "DataQualityTier",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PhaseProgress",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    # --- SME MRV Bridge ---
    "SMEMRVBridge",
    "SMEMRVBridgeConfig",
    "SMEMRVAgentRoute",
    "SMEEmissionSource",
    "SMEActivityData",
    "MRVScope",
    "RoutingResult",
    "BatchRoutingResult",
    "MRVValidationResult",
    # --- SME Data Bridge ---
    "SMEDataBridge",
    "SMEDataBridgeConfig",
    "SMEDataSourceType",
    "SMEDataCategory",
    "SMESpendCategory",
    "IntakeResult",
    "QualityResult",
    "SpendMappingResult",
    # --- Xero Connector ---
    "XeroConnector",
    "XeroConfig",
    "XeroConnectionStatus",
    "XeroAccount",
    "XeroTransaction",
    "XeroExportResult",
    "XeroAggregation",
    # --- QuickBooks Connector ---
    "QuickBooksConnector",
    "QBConfig",
    "QBConnectionStatus",
    "QBAccount",
    "QBTransaction",
    "QBExportResult",
    "QBAggregation",
    # --- Sage Connector ---
    "SageConnector",
    "SageConfig",
    "SageConnectionStatus",
    "SageNominalAccount",
    "SageTransaction",
    "SageExportResult",
    "SageAggregation",
    # --- Grant Database Bridge ---
    "GrantDatabaseBridge",
    "GrantDatabaseConfig",
    "GrantRegion",
    "GrantStatus",
    "GrantCategory",
    "ApplicationStatus",
    "GrantMatch",
    "GrantSearchResult",
    "GrantApplication",
    "DeadlineAlert",
    # --- SME Climate Hub Bridge ---
    "SMEClimateHubBridge",
    "SMEClimateHubConfig",
    "CommitmentStatus",
    "ProgressStatus",
    "BadgeType",
    "CommitmentData",
    "CommitmentResult",
    "ProgressReport",
    "ProgressSubmissionResult",
    "VerificationStatus",
    "BadgeInfo",
    # --- Certification Body Bridge ---
    "CertificationBodyBridge",
    "CertificationBodyConfig",
    "CertificationType",
    "CertificationStatus",
    "DocumentType",
    "CertificationSubmission",
    "DocumentUploadResult",
    "CertificationReadiness",
    # --- Peer Network Bridge ---
    "PeerNetworkBridge",
    "PeerNetworkConfig",
    "SizeTier",
    "BenchmarkMetric",
    "BenchmarkResult",
    "PercentileRanking",
    # --- Renewable PPA Marketplace ---
    "RenewablePPAMarketplace",
    "PPAMarketplaceConfig",
    "PPAProvider",
    "EnergySource",
    "PPAType",
    "PPAContract",
    "PPASearchResult",
    "CostComparison",
    "ComparisonTable",
    "PPAInterest",
    # --- SME Setup Wizard ---
    "SMESetupWizard",
    "SMEWizardStep",
    "StepStatus",
    "SMESize",
    "AccountingSoftware",
    "CertificationPathway",
    "SMEOrganizationProfile",
    "DataQualitySelection",
    "AccountingConnectionSetup",
    "GrantPreferences",
    "CertificationSelection",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- SME Health Check ---
    "SMEHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
]
