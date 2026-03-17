# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Integration Layer
=================================================

Phase 5 integration layer that extends PACK-004 CBAM Readiness with full
registry API connectivity, TARIC database integration, EU ETS Union Registry
bridging, cross-pack regulation synchronization, 10-step guided setup, and
18-category health verification. This module provides the production-grade
orchestration, external system connectivity, and operational readiness
validation for the CBAM Complete solution.

Components:
    - CBAMCompleteOrchestrator:   10-phase CBAM Complete execution pipeline
    - CBAMRegistryClient:         HTTP client for EU CBAM Registry APIs
    - TARICClient:                EU TARIC database integration
    - ETSRegistryBridge:          EU ETS Union Registry (EUTL) data integration
    - CrossPackBridge:            Bridge to other GreenLang Solution Packs
    - CBAMCompleteSetupWizard:    10-step interactive setup wizard
    - CBAMCompleteHealthCheck:    18-category health verification system

Architecture:
    Import Data --> CBAMCompleteOrchestrator --> PACK-004 Base Pipeline
                            |                           |
                            v                           v
    TARICClient --> CN Validation       ETSRegistryBridge --> Benchmark Data
                            |                           |
                            v                           v
    CBAMRegistryClient --> Declaration       CrossPackBridge --> CSRD/CDP/SBTi
                            |                           |
                            v                           v
    CBAMCompleteSetupWizard <-- Config       Audit Trail --> Provenance
                            |
                            v
    CBAMCompleteHealthCheck --> Readiness Verification

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-005"
__pack_name__ = "CBAM Complete Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.pack_orchestrator import (
    CBAMCompleteOrchestrator,
    CBAMCompleteConfig,
    CBAMCompletePhase,
    CheckpointData,
    CompleteExecutionStatus,
    OrchestrationResult,
    OrchestrationStatus,
    PhaseResult as CompletePhaseResult,
)

# ---------------------------------------------------------------------------
# Registry Client
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.registry_client import (
    AmendmentReceipt,
    AuthToken,
    BalanceResponse,
    CBAMRegistryClient,
    DeclarantStatus,
    FinalStatus,
    OAuthToken,
    PriceResponse,
    PurchaseReceipt,
    RegistrationReceipt,
    RegistryAPIConfig,
    RegistryError,
    ResaleReceipt,
    SubmissionReceipt,
    SubmissionStatus,
    SurrenderReceipt,
)

# ---------------------------------------------------------------------------
# TARIC Client
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.taric_client import (
    CBAMApplicability,
    CNCodeChange,
    CNCodeMatch,
    CNCodeValidation,
    CNHierarchy,
    CustomsAutomationConfig,
    DownstreamProduct,
    DutyRate,
    TARICClient,
    TariffMeasure,
)

# ---------------------------------------------------------------------------
# ETS Registry Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.ets_registry_bridge import (
    BenchmarkValue,
    ComplianceStatus as ETSComplianceStatus,
    ConsistencyCheck,
    CrossReference,
    ETSPrice as ETSRegistryPrice,
    ETSRegistryBridge,
    FreeAllocation,
    Installation,
    Sector,
)

# ---------------------------------------------------------------------------
# Cross-Pack Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.cross_pack_bridge import (
    CDPPushResult,
    CrossPackBridge,
    CrossRegulationConfig,
    CSRDPushResult,
    ETSPushResult,
    EUDRPushResult,
    PackAvailability,
    SBTiPushResult,
    SyncResult,
    TaxonomyPushResult,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.setup_wizard import (
    CBAMCompleteSetupWizard,
    SetupResult as CompleteSetupResult,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_005_cbam_complete.integrations.health_check import (
    CBAMCompleteHealthCheck,
    CategoryResult as CompleteCategoryResult,
    CompleteCheckCategory,
    CompleteFinding,
    CompleteHealthStatus,
    CompleteSeverity,
    HealthCheckResult as CompleteHealthCheckResult,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Pack Orchestrator
    "CBAMCompleteOrchestrator",
    "CBAMCompleteConfig",
    "CBAMCompletePhase",
    "CompleteExecutionStatus",
    "CompletePhaseResult",
    "OrchestrationResult",
    "OrchestrationStatus",
    "CheckpointData",
    # Registry Client
    "CBAMRegistryClient",
    "RegistryAPIConfig",
    "RegistryError",
    "AuthToken",
    "OAuthToken",
    "SubmissionReceipt",
    "AmendmentReceipt",
    "SubmissionStatus",
    "PurchaseReceipt",
    "SurrenderReceipt",
    "ResaleReceipt",
    "BalanceResponse",
    "PriceResponse",
    "RegistrationReceipt",
    "DeclarantStatus",
    "FinalStatus",
    # TARIC Client
    "TARICClient",
    "CustomsAutomationConfig",
    "CNCodeValidation",
    "CNHierarchy",
    "TariffMeasure",
    "CBAMApplicability",
    "CNCodeChange",
    "DownstreamProduct",
    "CNCodeMatch",
    "DutyRate",
    # ETS Registry Bridge
    "ETSRegistryBridge",
    "FreeAllocation",
    "BenchmarkValue",
    "ETSComplianceStatus",
    "CrossReference",
    "ConsistencyCheck",
    "ETSRegistryPrice",
    "Installation",
    "Sector",
    # Cross-Pack Bridge
    "CrossPackBridge",
    "CrossRegulationConfig",
    "CSRDPushResult",
    "CDPPushResult",
    "SBTiPushResult",
    "TaxonomyPushResult",
    "ETSPushResult",
    "EUDRPushResult",
    "PackAvailability",
    "SyncResult",
    # Setup Wizard
    "CBAMCompleteSetupWizard",
    "CompleteSetupResult",
    # Health Check
    "CBAMCompleteHealthCheck",
    "CompleteHealthCheckResult",
    "CompleteCategoryResult",
    "CompleteFinding",
    "CompleteCheckCategory",
    "CompleteHealthStatus",
    "CompleteSeverity",
]
