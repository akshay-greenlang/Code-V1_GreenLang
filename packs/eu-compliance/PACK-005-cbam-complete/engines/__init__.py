# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete - Computation Engines

Eight specialized engines providing the computational backbone for
enterprise-grade CBAM (Carbon Border Adjustment Mechanism) compliance:

    1. CertificateTradingEngine      - Certificate lifecycle management
                                       (purchase, surrender, resale, valuation)
    2. PrecursorChainEngine          - Multi-tier precursor emission resolution
                                       (6 goods categories, recursive chains)
    3. MultiEntityEngine             - Corporate group CBAM management
                                       (multi-EORI, 27 EU member states)
    4. RegistryAPIEngine             - EU CBAM Registry API integration
                                       (declarations, certificates, status)
    5. AdvancedAnalyticsEngine       - Strategic cost optimization
                                       (Monte Carlo, sourcing, scenarios)
    6. CustomsAutomationEngine       - Customs integration & anti-circumvention
                                       (CN validation, 5 detection rules)
    7. CrossRegulationEngine         - Multi-regulation data mapping
                                       (CSRD, CDP, SBTi, Taxonomy, ETS, EUDR)
    8. AuditManagementEngine         - Audit trail & NCA examination readiness
                                       (evidence, anomaly detection, penalties)

Regulatory Basis:
    EU Regulation 2023/956 (CBAM Regulation)
    EU Implementing Regulation 2023/1773 (Transitional period)
    EU Delegated Regulation 2023/1775 (Methodology)

Pack Tier: Enterprise (PACK-005)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

from __future__ import annotations

__version__: str = "1.0.0"
__pack__: str = "PACK-005"
__pack_name__: str = "CBAM Complete"
__engines_count__: int = 8

# ===================================================================
# Engine 1: Certificate Trading
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.certificate_trading_engine import (
    AlertSeverity,
    BudgetForecast,
    Certificate,
    CertificatePortfolio,
    CertificateStatus,
    CertificateTradingConfig,
    CertificateTradingEngine,
    ExecutionResult,
    ExpiryAlert,
    HoldingCheck,
    OrderStatus,
    PortfolioValuation,
    PurchaseOrder,
    ResaleResult,
    SurrenderPlan,
    SurrenderResult,
    TransferResult,
    ValuationMethod,
)

# ===================================================================
# Engine 2: Precursor Chain
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.precursor_chain_engine import (
    AllocationMethod,
    AllocationResult,
    ChainVisualization,
    CompositionRecord,
    DefaultFallbackResult,
    DefaultValueTier,
    GapAnalysis,
    GoodsCategory,
    MassBalanceResult,
    PrecursorChain,
    PrecursorChainConfig,
    PrecursorChainEngine,
    PrecursorEmissionResult,
    PrecursorNode,
    ProductionRoute,
    ProductionRouteType,
    ScrapClassification,
    ScrapType,
)

# ===================================================================
# Engine 3: Multi-Entity
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.multi_entity_engine import (
    ComplianceStatus,
    ConsolidatedObligation,
    CostAllocation,
    CostAllocationMethod,
    DelegatedComplianceStatus,
    Entity,
    EntityDeclaration,
    EntityGroup,
    EntityType,
    FinancialGuaranteeRecord,
    GroupDeMinimisResult,
    GuaranteeStatus,
    HierarchyResult,
    MemberStateCoordination,
    MultiEntityConfig,
    MultiEntityEngine,
    EU_MEMBER_STATES,
)

# ===================================================================
# Engine 4: Registry API
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.registry_api_engine import (
    AmendmentResult,
    APIErrorCode,
    AuditLogEntry,
    BalanceResult,
    DeclarantStatus,
    DeclarantStatusResult,
    FinalStatus,
    PriceResult,
    PurchaseConfirmation,
    RegistrationResult,
    RegistryAPIConfig,
    RegistryAPIEngine,
    RegistryAuth,
    RegistryOperationType,
    ResaleConfirmation,
    StatusCheckResult,
    SubmissionResult,
    SubmissionStatus,
    SurrenderConfirmation,
)

# ===================================================================
# Engine 5: Advanced Analytics
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.advanced_analytics_engine import (
    AdvancedAnalyticsConfig,
    AdvancedAnalyticsEngine,
    BenchmarkResult,
    BudgetProjection,
    DecarbROI,
    Distribution,
    MonteCarloResult,
    OptimizationObjective,
    PhaseOutImpact,
    PriceForecast,
    PriceTrend,
    ScenarioResults,
    SensitivityResult,
    SourcingOptimization,
    SupplierOption,
    TCOAnalysis,
)

# ===================================================================
# Engine 6: Customs Automation
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.customs_automation_engine import (
    AEOStatus,
    AEOStatusType,
    ApplicabilityResult,
    CBAMApplicability,
    CircumventionAlert,
    CircumventionType,
    CNValidationResult,
    CombinedCostResult,
    CustomsAutomationConfig,
    CustomsAutomationEngine,
    DownstreamMonitorResult,
    EORIValidation,
    ImportProcedureStatus,
    ParsedDeclaration,
    ProcedureCheck,
    VersionChanges,
)

# ===================================================================
# Engine 7: Cross-Regulation
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.cross_regulation_engine import (
    CARBON_PRICING_DB,
    CarbonPricingEquivalence,
    CDPMapping,
    ChangeTracker,
    ConsistencyCheckResult,
    CrossRegulationConfig,
    CrossRegulationEngine,
    CSRDMapping,
    DataReuseReport,
    ETSMapping,
    EUDRMapping,
    SBTiMapping,
    TaxonomyMapping,
)

# ===================================================================
# Engine 8: Audit Management
# ===================================================================
from packs.eu_compliance.PACK_005_cbam_complete.engines.audit_management_engine import (
    AccreditationStatus,
    AccreditationStatusResult,
    AnomalyAlert,
    AnomalySeverity,
    AuditCommitteeReport,
    AuditManagementConfig,
    AuditManagementEngine,
    AuditRepository,
    ChainOfCustody,
    CorrespondenceRecord,
    DataRoom,
    DataRoomAccess,
    EvidenceRecord,
    EvidenceType,
    ExaminationPackage,
    PenaltyExposure,
    RemediationPlan,
    RemediationPriority,
    RemediationStatus,
)

# ===================================================================
# Public API
# ===================================================================

__all__: list[str] = [
    # Pack metadata
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
    # Engine 1: Certificate Trading
    "CertificateTradingEngine",
    "CertificateTradingConfig",
    "Certificate",
    "CertificatePortfolio",
    "CertificateStatus",
    "PurchaseOrder",
    "OrderStatus",
    "ExecutionResult",
    "SurrenderResult",
    "ResaleResult",
    "ExpiryAlert",
    "AlertSeverity",
    "HoldingCheck",
    "PortfolioValuation",
    "ValuationMethod",
    "TransferResult",
    "SurrenderPlan",
    "BudgetForecast",
    # Engine 2: Precursor Chain
    "PrecursorChainEngine",
    "PrecursorChainConfig",
    "PrecursorNode",
    "PrecursorChain",
    "PrecursorEmissionResult",
    "AllocationResult",
    "AllocationMethod",
    "CompositionRecord",
    "DefaultFallbackResult",
    "DefaultValueTier",
    "MassBalanceResult",
    "ScrapClassification",
    "ScrapType",
    "ProductionRoute",
    "ProductionRouteType",
    "GapAnalysis",
    "ChainVisualization",
    "GoodsCategory",
    # Engine 3: Multi-Entity
    "MultiEntityEngine",
    "MultiEntityConfig",
    "Entity",
    "EntityGroup",
    "EntityType",
    "ConsolidatedObligation",
    "GroupDeMinimisResult",
    "CostAllocation",
    "CostAllocationMethod",
    "EntityDeclaration",
    "FinancialGuaranteeRecord",
    "GuaranteeStatus",
    "MemberStateCoordination",
    "DelegatedComplianceStatus",
    "ComplianceStatus",
    "HierarchyResult",
    "EU_MEMBER_STATES",
    # Engine 4: Registry API
    "RegistryAPIEngine",
    "RegistryAPIConfig",
    "RegistryAuth",
    "SubmissionResult",
    "SubmissionStatus",
    "AmendmentResult",
    "StatusCheckResult",
    "PurchaseConfirmation",
    "SurrenderConfirmation",
    "ResaleConfirmation",
    "BalanceResult",
    "PriceResult",
    "RegistrationResult",
    "DeclarantStatusResult",
    "DeclarantStatus",
    "FinalStatus",
    "RegistryOperationType",
    "APIErrorCode",
    "AuditLogEntry",
    # Engine 5: Advanced Analytics
    "AdvancedAnalyticsEngine",
    "AdvancedAnalyticsConfig",
    "SourcingOptimization",
    "SupplierOption",
    "OptimizationObjective",
    "ScenarioResults",
    "MonteCarloResult",
    "Distribution",
    "PriceForecast",
    "PriceTrend",
    "PhaseOutImpact",
    "DecarbROI",
    "BenchmarkResult",
    "TCOAnalysis",
    "BudgetProjection",
    "SensitivityResult",
    # Engine 6: Customs Automation
    "CustomsAutomationEngine",
    "CustomsAutomationConfig",
    "CNValidationResult",
    "ParsedDeclaration",
    "AEOStatus",
    "AEOStatusType",
    "ApplicabilityResult",
    "CBAMApplicability",
    "ProcedureCheck",
    "ImportProcedureStatus",
    "CircumventionAlert",
    "CircumventionType",
    "DownstreamMonitorResult",
    "CombinedCostResult",
    "EORIValidation",
    "VersionChanges",
    # Engine 7: Cross-Regulation
    "CrossRegulationEngine",
    "CrossRegulationConfig",
    "CSRDMapping",
    "CDPMapping",
    "SBTiMapping",
    "TaxonomyMapping",
    "ETSMapping",
    "EUDRMapping",
    "DataReuseReport",
    "ConsistencyCheckResult",
    "ChangeTracker",
    "CarbonPricingEquivalence",
    "CARBON_PRICING_DB",
    # Engine 8: Audit Management
    "AuditManagementEngine",
    "AuditManagementConfig",
    "AuditRepository",
    "EvidenceRecord",
    "EvidenceType",
    "ChainOfCustody",
    "DataRoom",
    "DataRoomAccess",
    "RemediationPlan",
    "RemediationStatus",
    "RemediationPriority",
    "ExaminationPackage",
    "AnomalyAlert",
    "AnomalySeverity",
    "PenaltyExposure",
    "AuditCommitteeReport",
    "AccreditationStatusResult",
    "AccreditationStatus",
    "CorrespondenceRecord",
]
