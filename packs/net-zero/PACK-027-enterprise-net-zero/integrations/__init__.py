# -*- coding: utf-8 -*-
"""
PACK-027 Enterprise Net Zero Pack - Integration Layer
==========================================================

Phase 4 integration layer for the Enterprise Net Zero Pack providing
10-phase DAG pipeline orchestration, full 30-agent MRV routing, full
20-agent DATA routing, SAP S/4HANA connector (OData V4/BAPI/RFC),
Oracle ERP Cloud connector, Workday HCM connector, CDP Climate Change
questionnaire auto-population, SBTi Corporate Standard target validation
(42 criteria), Big 4 assurance provider integration, 100+ entity
hierarchy management, voluntary carbon credit marketplace, 100,000+
supplier engagement portal, general ledger carbon allocation, automated
data quality guardian, 8-step enterprise setup wizard, and 16-category
health monitoring.

Components:
    - EnterpriseNetZeroPipelineOrchestrator: 10-phase DAG pipeline
      (Onboarding -> Data Integration -> Entity Consolidation ->
      Enterprise Baseline -> DQ Assurance -> Target Setting ->
      Scenario Modeling -> Carbon Pricing -> Supply Chain -> Reporting)
    - EnterpriseMRVBridge: Routes to all 30 MRV agents at activity-based
      precision for financial-grade accuracy (+/-3%)
    - EnterpriseDataBridge: Full 20-agent DATA integration with ERP-grade
      extraction, cross-source reconciliation, and data lineage tracking
    - SAPConnector: SAP S/4HANA integration with OData V4, BAPI/RFC,
      multi-company-code support, and carbon cost write-back
    - OracleConnector: Oracle ERP Cloud integration with REST API V2,
      multi-business-unit support, and GL journal write-back
    - WorkdayConnector: Workday HCM integration for employee headcount,
      commuting data, and business travel (Cat 6/7)
    - CDPBridge: Automated CDP Climate Change questionnaire response
      with A-list targeting and score optimization
    - SBTiBridge: SBTi Corporate Standard target validation covering
      42 criteria (28 near-term C1-C28 + 14 net-zero NZ-C1 to NZ-C14)
    - AssuranceProviderBridge: Big 4 assurance provider integration with
      11 workpaper types, ISO 14064-3/ISAE 3410 compliance
    - MultiEntityOrchestrator: 100+ entity hierarchy management with
      3 consolidation approaches and base year recalculation
    - CarbonMarketplaceBridge: Voluntary carbon credit procurement with
      multi-registry search, VCMI Claims Code compliance
    - SupplyChainPortal: 100,000+ supplier engagement portal with
      4-tier model, scorecards, and hotspot analysis
    - FinancialSystemBridge: Internal carbon pricing and GL allocation
      with carbon-adjusted P&L, NPV, and CBAM exposure
    - DataQualityGuardian: 5-dimension data quality assessment with
      GHG Protocol DQ hierarchy and anomaly detection
    - EnterpriseSetupWizard: 8-step enterprise onboarding wizard
    - EnterpriseHealthCheck: 16-category system health monitoring

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-027 Enterprise <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (all 30) <-- DATA Agents (all 20) <-- ERP Systems
                              |
                              v
    SBTi/CDP/Assurance <-- Financial System <-- Supply Chain Portal
                              |
                              v
    Carbon Marketplace <-- Multi-Entity <-- Data Quality Guardian

Platform Integrations:
    - greenlang/agents/mrv/* (all 30 MRV agents)
    - greenlang/agents/data/* (all 20 DATA agents)
    - SAP S/4HANA API (OData V4, BAPI/RFC)
    - Oracle ERP Cloud API (REST V2)
    - Workday HCM API (REST, RaaS)
    - CDP Reporter Services API
    - SBTi Submission Portal API
    - Carbon credit registries (Verra, Gold Standard, ACR, CAR, Puro.earth)
    - Big 4 assurance portals (Deloitte, EY, KPMG, PwC)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-027"
__pack_name__ = "Enterprise Net Zero Pack"

# ---------------------------------------------------------------------------
# Enterprise Net Zero Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    EnterprisePipelinePhase,
    EnterprisePathType,
    ExecutionStatus,
    RetryConfig,
    EnterpriseOrchestratorConfig,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    PhaseProgress,
    EnterpriseNetZeroPipelineOrchestrator,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
)

# ---------------------------------------------------------------------------
# Enterprise MRV Bridge
# ---------------------------------------------------------------------------
from .mrv_bridge import (
    MRVScope,
    EnterpriseMRVAgentRoute,
    EnterpriseMRVBridgeConfig,
    RoutingResult,
    BatchRoutingResult,
    EnterpriseMRVBridge,
    ENTERPRISE_MRV_ROUTING_TABLE,
)

# ---------------------------------------------------------------------------
# Enterprise Data Bridge
# ---------------------------------------------------------------------------
from .data_bridge import (
    EnterpriseDataBridgeConfig,
    IntakeResult,
    ReconciliationResult,
    EnterpriseDataBridge,
    ENTERPRISE_DATA_AGENT_ROUTING,
)

# ---------------------------------------------------------------------------
# SAP Connector
# ---------------------------------------------------------------------------
from .sap_connector import (
    SAPModule,
    SAPAuthMethod,
    SAPConnectionProtocol,
    ExtractionMode,
    ScopeMapping,
    SAPConfig,
    SAPConnectionStatus,
    SAPExtractionRequest,
    SAPExtractionResult,
    SAPMaterialGroupMapping,
    SAPWriteBackRequest,
    SAPWriteBackResult,
    SAPConnector,
    DEFAULT_MATERIAL_GROUP_MAPPINGS,
)

# ---------------------------------------------------------------------------
# Oracle Connector
# ---------------------------------------------------------------------------
from .oracle_connector import (
    OracleModule,
    OracleAuthMethod,
    OracleExtractionMode,
    OracleConfig,
    OracleConnectionStatus,
    OracleExtractionResult,
    OracleWriteBackResult,
    OracleConnector,
    COMMODITY_SCOPE3_MAP,
)

# ---------------------------------------------------------------------------
# Workday Connector
# ---------------------------------------------------------------------------
from .workday_connector import (
    WorkdayDataType,
    WorkArrangement,
    CommuteMode,
    WorkdayConfig,
    WorkdayConnectionStatus,
    WorkdayExtractionResult,
    HeadcountByLocation,
    TravelSummary,
    WorkdayConnector,
)

# ---------------------------------------------------------------------------
# CDP Bridge
# ---------------------------------------------------------------------------
from .cdp_bridge import (
    CDPModule,
    CDPScore,
    CDPSubmissionStatus,
    CDPBridgeConfig,
    CDPModuleResponse,
    CDPPopulationResult,
    CDPSubmissionResult,
    CDPBridge,
    CDP_MODULE_INFO,
)

# ---------------------------------------------------------------------------
# SBTi Bridge
# ---------------------------------------------------------------------------
from .sbti_bridge import (
    SBTiPathway,
    SBTiTargetType,
    CriteriaStatus,
    SBTiSubmissionStatus,
    TemperatureRating,
    SBTiBridgeConfig,
    CriteriaValidation,
    SBTiTargetDefinition,
    SBTiValidationResult,
    SBTiProgressReport,
    SBTiBridge,
    SBTI_NEAR_TERM_CRITERIA,
    SBTI_NET_ZERO_CRITERIA,
)

# ---------------------------------------------------------------------------
# Assurance Provider Bridge
# ---------------------------------------------------------------------------
from .assurance_provider_bridge import (
    AssuranceLevel,
    AssuranceStandard,
    AssuranceProvider,
    WorkpaperType,
    EngagementStatus,
    AssuranceBridgeConfig,
    Workpaper,
    AssurancePackage,
    AssuranceProviderBridge,
)

# ---------------------------------------------------------------------------
# Multi-Entity Orchestrator
# ---------------------------------------------------------------------------
from .multi_entity_orchestrator import (
    ConsolidationApproach,
    EntityType,
    StructuralChangeType,
    MultiEntityConfig,
    EntityDefinition,
    IntercompanyTransaction,
    StructuralChange,
    ConsolidationResult,
    MultiEntityOrchestrator,
)

# ---------------------------------------------------------------------------
# Carbon Marketplace Bridge
# ---------------------------------------------------------------------------
from .carbon_marketplace_bridge import (
    CreditRegistry,
    CreditType,
    CreditStatus,
    VCMITier,
    CarbonMarketplaceConfig,
    CarbonCredit,
    CreditSearchResult,
    CreditPurchaseResult,
    CreditRetirementResult,
    PortfolioSummary,
    CarbonMarketplaceBridge,
)

# ---------------------------------------------------------------------------
# Supply Chain Portal
# ---------------------------------------------------------------------------
from .supply_chain_portal import (
    SupplierTier,
    EngagementStage,
    QuestionnaireStatus,
    SupplyChainPortalConfig,
    Supplier,
    SupplierScorecard,
    HotspotAnalysis,
    EngagementProgress,
    SupplyChainPortal,
)

# Avoid collision with cdp_bridge.CDPScore
from .supply_chain_portal import CDPScore as SupplierCDPScore

# ---------------------------------------------------------------------------
# Financial System Bridge
# ---------------------------------------------------------------------------
from .financial_system_bridge import (
    CarbonPricingApproach,
    AllocationMethod,
    FinancialBridgeConfig,
    CostCenterAllocation,
    CarbonAdjustedPL,
    InvestmentAppraisal,
    CBAMExposure,
    FinancialSystemBridge,
)

# ---------------------------------------------------------------------------
# Data Quality Guardian
# ---------------------------------------------------------------------------
from .data_quality_guardian import (
    DQDimension,
    DQLevel,
    AnomalyType,
    DQSeverity,
    DataQualityGuardianConfig,
    DQIssue,
    DQDimensionScore,
    DQAssessmentResult,
    YoYVarianceResult,
    DataQualityGuardian,
)

# ---------------------------------------------------------------------------
# Enterprise Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    EnterpriseWizardStep,
    StepStatus,
    ERPSystem,
    SectorPreset,
    EnterpriseOrgProfile,
    EntityHierarchySetup,
    ERPSetup,
    WizardStepState,
    WizardState,
    EnterpriseSetupResult,
    EnterpriseSetupWizard,
    STEP_ORDER,
    STEP_DISPLAY_NAMES,
)

# ---------------------------------------------------------------------------
# Enterprise Health Check
# ---------------------------------------------------------------------------
from .health_check import (
    HealthStatus,
    HealthSeverity,
    CheckCategory,
    RemediationSuggestion,
    ComponentHealth,
    HealthCheckConfig,
    HealthCheckResult,
    EnterpriseHealthCheck,
    ENTERPRISE_ENGINES,
    ENTERPRISE_WORKFLOWS,
    ENTERPRISE_TEMPLATES,
    QUICK_CHECK_CATEGORIES,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Enterprise Pipeline Orchestrator ---
    "EnterpriseNetZeroPipelineOrchestrator",
    "EnterpriseOrchestratorConfig",
    "RetryConfig",
    "EnterprisePipelinePhase",
    "EnterprisePathType",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PhaseProgress",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    # --- Enterprise MRV Bridge ---
    "EnterpriseMRVBridge",
    "EnterpriseMRVBridgeConfig",
    "EnterpriseMRVAgentRoute",
    "MRVScope",
    "RoutingResult",
    "BatchRoutingResult",
    "ENTERPRISE_MRV_ROUTING_TABLE",
    # --- Enterprise Data Bridge ---
    "EnterpriseDataBridge",
    "EnterpriseDataBridgeConfig",
    "IntakeResult",
    "ReconciliationResult",
    "ENTERPRISE_DATA_AGENT_ROUTING",
    # --- SAP Connector ---
    "SAPConnector",
    "SAPConfig",
    "SAPModule",
    "SAPAuthMethod",
    "SAPConnectionProtocol",
    "ExtractionMode",
    "ScopeMapping",
    "SAPConnectionStatus",
    "SAPExtractionRequest",
    "SAPExtractionResult",
    "SAPMaterialGroupMapping",
    "SAPWriteBackRequest",
    "SAPWriteBackResult",
    "DEFAULT_MATERIAL_GROUP_MAPPINGS",
    # --- Oracle Connector ---
    "OracleConnector",
    "OracleConfig",
    "OracleModule",
    "OracleAuthMethod",
    "OracleExtractionMode",
    "OracleConnectionStatus",
    "OracleExtractionResult",
    "OracleWriteBackResult",
    "COMMODITY_SCOPE3_MAP",
    # --- Workday Connector ---
    "WorkdayConnector",
    "WorkdayConfig",
    "WorkdayDataType",
    "WorkArrangement",
    "CommuteMode",
    "WorkdayConnectionStatus",
    "WorkdayExtractionResult",
    "HeadcountByLocation",
    "TravelSummary",
    # --- CDP Bridge ---
    "CDPBridge",
    "CDPBridgeConfig",
    "CDPModule",
    "CDPScore",
    "CDPSubmissionStatus",
    "CDPModuleResponse",
    "CDPPopulationResult",
    "CDPSubmissionResult",
    "CDP_MODULE_INFO",
    # --- SBTi Bridge ---
    "SBTiBridge",
    "SBTiBridgeConfig",
    "SBTiPathway",
    "SBTiTargetType",
    "CriteriaStatus",
    "SBTiSubmissionStatus",
    "TemperatureRating",
    "CriteriaValidation",
    "SBTiTargetDefinition",
    "SBTiValidationResult",
    "SBTiProgressReport",
    "SBTI_NEAR_TERM_CRITERIA",
    "SBTI_NET_ZERO_CRITERIA",
    # --- Assurance Provider Bridge ---
    "AssuranceProviderBridge",
    "AssuranceBridgeConfig",
    "AssuranceLevel",
    "AssuranceStandard",
    "AssuranceProvider",
    "WorkpaperType",
    "EngagementStatus",
    "Workpaper",
    "AssurancePackage",
    # --- Multi-Entity Orchestrator ---
    "MultiEntityOrchestrator",
    "MultiEntityConfig",
    "ConsolidationApproach",
    "EntityType",
    "StructuralChangeType",
    "EntityDefinition",
    "IntercompanyTransaction",
    "StructuralChange",
    "ConsolidationResult",
    # --- Carbon Marketplace Bridge ---
    "CarbonMarketplaceBridge",
    "CarbonMarketplaceConfig",
    "CreditRegistry",
    "CreditType",
    "CreditStatus",
    "VCMITier",
    "CarbonCredit",
    "CreditSearchResult",
    "CreditPurchaseResult",
    "CreditRetirementResult",
    "PortfolioSummary",
    # --- Supply Chain Portal ---
    "SupplyChainPortal",
    "SupplyChainPortalConfig",
    "SupplierTier",
    "EngagementStage",
    "QuestionnaireStatus",
    "SupplierCDPScore",
    "Supplier",
    "SupplierScorecard",
    "HotspotAnalysis",
    "EngagementProgress",
    # --- Financial System Bridge ---
    "FinancialSystemBridge",
    "FinancialBridgeConfig",
    "CarbonPricingApproach",
    "AllocationMethod",
    "CostCenterAllocation",
    "CarbonAdjustedPL",
    "InvestmentAppraisal",
    "CBAMExposure",
    # --- Data Quality Guardian ---
    "DataQualityGuardian",
    "DataQualityGuardianConfig",
    "DQDimension",
    "DQLevel",
    "AnomalyType",
    "DQSeverity",
    "DQIssue",
    "DQDimensionScore",
    "DQAssessmentResult",
    "YoYVarianceResult",
    # --- Enterprise Setup Wizard ---
    "EnterpriseSetupWizard",
    "EnterpriseWizardStep",
    "StepStatus",
    "ERPSystem",
    "SectorPreset",
    "EnterpriseOrgProfile",
    "EntityHierarchySetup",
    "ERPSetup",
    "WizardStepState",
    "WizardState",
    "EnterpriseSetupResult",
    "STEP_ORDER",
    "STEP_DISPLAY_NAMES",
    # --- Enterprise Health Check ---
    "EnterpriseHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    "ENTERPRISE_ENGINES",
    "ENTERPRISE_WORKFLOWS",
    "ENTERPRISE_TEMPLATES",
    "QUICK_CHECK_CATEGORIES",
]
