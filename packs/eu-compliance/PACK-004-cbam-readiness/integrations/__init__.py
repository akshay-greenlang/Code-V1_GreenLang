# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Integration Layer
==================================================

Phase 4 integration layer that connects GreenLang agents, GL-CBAM-APP v1.1,
customs data sources, and EU ETS price feeds into a cohesive CBAM compliance
pipeline. This module provides orchestration, bridging, CN code lookups,
ETS price integration, guided setup, and health verification for production
deployments.

Components:
    - CBAMPackOrchestrator:  8-phase CBAM execution pipeline (quarterly & annual)
    - CBAMAppBridge:         Bridge to GL-CBAM-APP v1.1 engines and workflows
    - CustomsBridge:         CN code database, EORI validation, customs parsing
    - ETSBridge:             EU ETS price feed, certificate price calculation
    - CBAMSetupWizard:       7-step guided setup for new deployments
    - CBAMHealthCheck:       12-category health verification system

Architecture:
    Import Data --> CBAMPackOrchestrator --> Validation --> Supplier Data
                            |                                    |
                            v                                    v
    CustomsBridge --> CN Code Lookup            CBAMAppBridge --> Emission Calc
                            |                                    |
                            v                                    v
    ETSBridge --> Certificate Price         Policy Check --> Report Generation
                            |                                    |
                            v                                    v
    CBAMSetupWizard <-- Configuration          Audit Trail --> Provenance

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from packs.eu_compliance.PACK_004_cbam_readiness.integrations.pack_orchestrator import (
    AnnualExecutionResult,
    CBAMOrchestratorConfig,
    CBAMPackOrchestrator,
    CBAMPhase,
    Checkpoint,
    ExecutionStatus,
    PhaseResult,
    QualityGateStatus,
    QuarterlyExecutionResult,
)
from packs.eu_compliance.PACK_004_cbam_readiness.integrations.cbam_app_bridge import (
    AppHealthStatus,
    CBAMAppBridge,
    CertificateEngineProxy,
    CNCodeEntry,
    ComplianceRule,
    CountryCarbonPrice,
    DeMinimisProxy,
    EmissionCalculatorProxy,
    EmissionFactorEntry,
    QuarterlyEngineProxy,
    SupplierPortalProxy,
    VerificationProxy,
)
from packs.eu_compliance.PACK_004_cbam_readiness.integrations.customs_bridge import (
    CNCodeInfo,
    CountryInfo,
    CustomsBridge,
    CustomsBridgeResult,
    CustomsDeclarationItem,
    GoodsCategory,
)
from packs.eu_compliance.PACK_004_cbam_readiness.integrations.ets_bridge import (
    CarbonPriceComparison,
    CertificatePriceResult,
    Currency,
    ETSBridge,
    ETSPrice,
    PriceSource,
    ProjectionScenario,
)
from packs.eu_compliance.PACK_004_cbam_readiness.integrations.setup_wizard import (
    CBAMSetupWizard,
    CNCodeConfig,
    CNCodeMapping,
    CompanyProfile,
    DataSourceConfig,
    DataSourceEntry,
    GoodsCategorySelection,
    ReportingPreferences,
    SetupResult,
    StepStatus,
    SupplierEntry,
    SupplierRegistry,
    WizardStep,
    WizardStepName,
)
from packs.eu_compliance.PACK_004_cbam_readiness.integrations.health_check import (
    CBAMHealthCheck,
    CategoryResult,
    CheckCategory,
    Finding,
    HealthCheckResult,
    HealthStatus,
    Severity,
)

__all__ = [
    # Pack Orchestrator
    "CBAMPackOrchestrator",
    "CBAMOrchestratorConfig",
    "CBAMPhase",
    "ExecutionStatus",
    "QualityGateStatus",
    "PhaseResult",
    "QuarterlyExecutionResult",
    "AnnualExecutionResult",
    "Checkpoint",
    # CBAM App Bridge
    "CBAMAppBridge",
    "AppHealthStatus",
    "CertificateEngineProxy",
    "QuarterlyEngineProxy",
    "SupplierPortalProxy",
    "DeMinimisProxy",
    "VerificationProxy",
    "EmissionCalculatorProxy",
    "CNCodeEntry",
    "EmissionFactorEntry",
    "ComplianceRule",
    "CountryCarbonPrice",
    # Customs Bridge
    "CustomsBridge",
    "CNCodeInfo",
    "CountryInfo",
    "CustomsDeclarationItem",
    "CustomsBridgeResult",
    "GoodsCategory",
    # ETS Bridge
    "ETSBridge",
    "ETSPrice",
    "PriceSource",
    "ProjectionScenario",
    "Currency",
    "CarbonPriceComparison",
    "CertificatePriceResult",
    # Setup Wizard
    "CBAMSetupWizard",
    "CompanyProfile",
    "GoodsCategorySelection",
    "CNCodeMapping",
    "CNCodeConfig",
    "SupplierEntry",
    "SupplierRegistry",
    "DataSourceEntry",
    "DataSourceConfig",
    "ReportingPreferences",
    "SetupResult",
    "WizardStep",
    "WizardStepName",
    "StepStatus",
    # Health Check
    "CBAMHealthCheck",
    "HealthCheckResult",
    "CategoryResult",
    "Finding",
    "CheckCategory",
    "HealthStatus",
    "Severity",
]
