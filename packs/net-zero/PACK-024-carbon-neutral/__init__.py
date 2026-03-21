# -*- coding: utf-8 -*-
"""
PACK-024: Carbon Neutral Pack
=============================================================================

Comprehensive GreenLang deployment pack for carbon neutrality management.
Provides 10 engines covering GHG footprint quantification (ISO 14064-1),
carbon management planning with reduction-first strategy, 12-dimension
ICVCM CCP credit quality scoring, carbon credit portfolio optimization
with avoidance/removal and nature/tech allocation, registry retirement
tracking across 6 registries (Verra, Gold Standard, ACR, CAR, Puro.earth,
Isometric), neutralization balance per ISO 14068-1:2023 and PAS 2060:2014,
claims substantiation with VCMI Claims Code compliance, ISAE 3410
verification package assembly, annual cycle management, and permanence
risk assessment.

Supports 8 neutrality types (Corporate, SME, Event, Product, Building,
Service, Project, Portfolio) with tailored presets and cross-framework
reporting (ISO 14068-1, PAS 2060, GHG Protocol, VCMI Claims Code).

Components:
    Engines (10):
        - FootprintQuantificationEngine    (ISO 14064-1 GHG quantification)
        - CarbonMgmtPlanEngine             (Reduction-first MACC planning)
        - CreditQualityEngine              (12-dimension ICVCM CCP scoring)
        - PortfolioOptimizationEngine       (Markowitz-inspired allocation)
        - RegistryRetirementEngine         (6-registry retirement tracking)
        - NeutralizationBalanceEngine      (ISO 14068-1 / PAS 2060 balance)
        - ClaimsSubstantiationEngine       (VCMI Claims Code validation)
        - VerificationPackageEngine        (ISAE 3410 evidence assembly)
        - AnnualCycleEngine                (Multi-year cycle management)
        - PermanenceRiskEngine             (Buffer pool risk assessment)

    Workflows (8):
        - FullAnnualCycleWorkflow          (10-phase end-to-end)
        - FootprintAssessmentWorkflow      (4-phase quantification)
        - CarbonMgmtPlanWorkflow           (5-phase management plan)
        - CreditProcurementWorkflow        (4-phase credit procurement)
        - RetirementWorkflow               (3-phase registry retirement)
        - NeutralizationWorkflow           (5-phase balance calculation)
        - ClaimsValidationWorkflow         (4-phase claims validation)
        - VerificationWorkflow             (4-phase ISAE 3410 package)

    Templates (10):
        - FootprintReportTemplate
        - CarbonMgmtPlanReportTemplate
        - CreditPortfolioReportTemplate
        - RegistryRetirementReportTemplate
        - NeutralizationStatementReportTemplate
        - ClaimsSubstantiationReportTemplate
        - VerificationPackageReportTemplate
        - AnnualReportTemplate
        - PermanenceAssessmentReportTemplate
        - PublicDisclosureReportTemplate

    Integrations (12):
        - CarbonNeutralOrchestrator        (10-phase DAG pipeline)
        - CarbonNeutralMRVBridge           (30 MRV agents)
        - CarbonNeutralGHGAppBridge        (GL-GHG-APP v1.0)
        - CarbonNeutralDecarbBridge        (DECARB agents)
        - CarbonNeutralDataBridge          (20 DATA agents)
        - CarbonNeutralRegistryBridge      (6-registry API bridge)
        - CarbonNeutralCreditMarketplaceBridge  (marketplace integration)
        - CarbonNeutralVerificationBodyBridge   (ISAE 3410 bodies)
        - Pack021Bridge                    (optional PACK-021 bridge)
        - Pack023Bridge                    (optional PACK-023 bridge)
        - CarbonNeutralHealthCheck         (20-category verification)
        - CarbonNeutralSetupWizard         (6-step configuration)

    Presets (8):
        - corporate_neutrality             (Full org Scope 1+2+3)
        - sme_neutrality                   (Simplified SME Scope 1+2)
        - event_neutrality                 (Conference/event neutrality)
        - product_neutrality               (ISO 14067 product LCA)
        - building_neutrality              (CRREM building operations)
        - service_neutrality               (Office/cloud/digital)
        - project_neutrality               (ISO 14064-2 construction)
        - portfolio_neutrality             (Multi-entity consolidation)

Agent Dependencies:
    - 30 AGENT-MRV agents (Scope 1/2/3 emissions quantification)
    - 20 AGENT-DATA agents (data intake and quality management)
    - 10 AGENT-FOUND agents (platform foundation services)

Regulatory Framework:
    Primary:
        - ISO 14068-1:2023 (Carbon neutrality)
        - PAS 2060:2014 (Demonstrating carbon neutrality)
    Secondary:
        - ISO 14064-1:2018, ISO 14064-2:2019, ISO 14067:2018
        - GHG Protocol Corporate/Scope 2/Scope 3 Standards
        - ICVCM Core Carbon Principles (2023)
        - VCMI Claims Code of Practice (2023)
        - ISAE 3410, IPCC AR6, Oxford Principles, CRREM, PCAF

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-024"
__pack_name__ = "Carbon Neutral Pack"
__author__ = "GreenLang Platform Team"
__category__: str = "net-zero"

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__author__",
    "__category__",
]
