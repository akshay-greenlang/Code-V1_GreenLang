# -*- coding: utf-8 -*-
"""
PACK-030: Net Zero Reporting Pack
=============================================================================

Comprehensive GreenLang deployment pack for multi-framework climate
disclosure reporting. Provides 10 engines covering data aggregation from
multiple sources with reconciliation, AI-assisted narrative generation
with full citation chains, cross-framework metric mapping across 7
frameworks (SBTi/CDP/TCFD/GRI/ISSB/SEC/CSRD), XBRL/iXBRL generation for
SEC and CSRD digital filing, interactive HTML5 dashboard generation,
ISAE 3410 assurance evidence bundle packaging, multi-section report assembly
with cross-referencing, schema/completeness/consistency validation, multi-
language translation (EN/DE/FR/ES) with regulatory terminology preservation,
and multi-format rendering (PDF/HTML/Excel/JSON/XBRL).

Supports simultaneous reporting across all 7 major climate disclosure
frameworks with a single data collection pass, ensuring consistency and
reducing reporting burden by 60-80%.

Components:
    Engines (10):
        - DataAggregationEngine            (Multi-source data reconciliation)
        - NarrativeGenerationEngine        (AI-assisted narrative w/ citations)
        - FrameworkMappingEngine           (7-framework metric mapping)
        - XBRLTaggingEngine                (XBRL/iXBRL for SEC & CSRD)
        - DashboardGenerationEngine        (Interactive HTML5 dashboards)
        - AssurancePackagingEngine         (ISAE 3410 evidence bundles)
        - ReportCompilationEngine          (Multi-section report assembly)
        - ValidationEngine                 (Schema/completeness validation)
        - TranslationEngine                (EN/DE/FR/ES translation)
        - FormatRenderingEngine            (PDF/HTML/Excel/JSON/XBRL)

    Workflows (8):
        - SBTiProgressWorkflow                   (4 phases)
        - CDPQuestionnaireWorkflow               (5 phases)
        - TCFDDisclosureWorkflow                 (4 phases)
        - GRI305Workflow                         (3 phases)
        - IFRSS2Workflow                         (4 phases)
        - SECClimateWorkflow                     (5 phases)
        - CSRDESRSE1Workflow                     (4 phases)
        - MultiFrameworkWorkflow                 (7 phases)

    Templates (15):
        - SBTiProgressTemplate
        - CDPGovernanceTemplate
        - CDPEmissionsTemplate
        - TCFDGovernanceTemplate
        - TCFDStrategyTemplate
        - TCFDRiskTemplate
        - TCFDMetricsTemplate
        - GRI305Template
        - ISSBS2Template
        - SECClimateTemplate
        - CSRDE1Template
        - InvestorDashboardTemplate
        - RegulatorDashboardTemplate
        - CustomerCarbonTemplate
        - AssuranceEvidenceTemplate

    Integrations (12):
        - PACK021Integration               (Baseline/target import)
        - PACK022Integration               (Acceleration data import)
        - PACK028Integration               (Sector pathway data)
        - PACK029Integration               (Interim targets data)
        - GLSBTiAppIntegration             (GL-SBTi-APP sync)
        - GLCDPAppIntegration              (GL-CDP-APP sync)
        - GLTCFDAppIntegration             (GL-TCFD-APP sync)
        - GLGHGAppIntegration              (GL-GHG-APP sync)
        - XBRLTaxonomyIntegration          (XBRL taxonomy service)
        - TranslationIntegration           (Translation service)
        - OrchestratorIntegration          (DAG pipeline)
        - HealthCheckIntegration           (System health)

    Presets (8):
        - sbti_disclosure          (SBTi progress reporting)
        - cdp_questionnaire        (CDP Climate Change)
        - tcfd_disclosure          (TCFD 4-pillar reporting)
        - gri_305                  (GRI 305 emissions)
        - issb_s2                  (ISSB IFRS S2 climate)
        - sec_climate              (SEC Reg S-K climate)
        - csrd_e1                  (CSRD ESRS E1 climate)
        - multi_framework          (All frameworks combined)

Agent Dependencies:
    - 30 AGENT-MRV agents (Scope 1/2/3 emissions quantification)
    - 20 AGENT-DATA agents (data intake and quality management)
    - 10 AGENT-FOUND agents (platform foundation services)

Regulatory Framework:
    Primary:
        - SBTi Corporate Net-Zero Standard v1.2 (2024)
        - CDP Climate Change Questionnaire (2024)
        - TCFD Recommendations (2017, updated 2023)
        - GRI 305 (2016) -- Emissions disclosures
        - ISSB IFRS S2 (2023) -- Climate-related disclosures
        - SEC Climate Disclosure Rules (2024) -- Reg S-K
        - CSRD ESRS E1 (2024) -- Climate change
    Secondary:
        - GHG Protocol Corporate Standard (2004, revised 2015)
        - GHG Protocol Scope 3 Standard (2011)
        - ISO 14064-1:2018 -- Organizational GHG inventories
        - ISAE 3410 -- Assurance on GHG statements
        - XBRL International Specification 2.1

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-030"
__pack_name__ = "Net Zero Reporting Pack"
__author__ = "GreenLang Platform Team"
__category__: str = "net-zero"

# ---------------------------------------------------------------------------
# Engines (10)
# ---------------------------------------------------------------------------
from .engines import (
    DataAggregationEngine,
    NarrativeGenerationEngine,
    FrameworkMappingEngine,
    XBRLTaggingEngine,
    DashboardGenerationEngine,
    AssurancePackagingEngine,
    ReportCompilationEngine,
    ValidationEngine,
    TranslationEngine,
    FormatRenderingEngine,
)

# ---------------------------------------------------------------------------
# Workflows (8)
# ---------------------------------------------------------------------------
from .workflows import (
    SBTiProgressWorkflow,
    CDPQuestionnaireWorkflow,
    TCFDDisclosureWorkflow,
    GRI305Workflow,
    IFRSS2Workflow,
    SECClimateWorkflow,
    CSRDESRSE1Workflow,
    MultiFrameworkWorkflow,
)

# ---------------------------------------------------------------------------
# Templates (15 + Registry)
# ---------------------------------------------------------------------------
from .templates import (
    SBTiProgressTemplate,
    CDPGovernanceTemplate,
    CDPEmissionsTemplate,
    TCFDGovernanceTemplate,
    TCFDStrategyTemplate,
    TCFDRiskTemplate,
    TCFDMetricsTemplate,
    GRI305Template,
    ISSBS2Template,
    SECClimateTemplate,
    CSRDE1Template,
    InvestorDashboardTemplate,
    RegulatorDashboardTemplate,
    CustomerCarbonTemplate,
    AssuranceEvidenceTemplate,
    TemplateRegistry,
)

# ---------------------------------------------------------------------------
# Integrations (12)
# ---------------------------------------------------------------------------
from .integrations import (
    PACK021Integration,
    PACK022Integration,
    PACK028Integration,
    PACK029Integration,
    GLSBTiAppIntegration,
    GLCDPAppIntegration,
    GLTCFDAppIntegration,
    GLGHGAppIntegration,
    XBRLTaxonomyIntegration,
    TranslationIntegration,
    OrchestratorIntegration,
    HealthCheckIntegration,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__author__",
    # --- Engines (10) ---
    "DataAggregationEngine",
    "NarrativeGenerationEngine",
    "FrameworkMappingEngine",
    "XBRLTaggingEngine",
    "DashboardGenerationEngine",
    "AssurancePackagingEngine",
    "ReportCompilationEngine",
    "ValidationEngine",
    "TranslationEngine",
    "FormatRenderingEngine",
    # --- Workflows (8) ---
    "SBTiProgressWorkflow",
    "CDPQuestionnaireWorkflow",
    "TCFDDisclosureWorkflow",
    "GRI305Workflow",
    "IFRSS2Workflow",
    "SECClimateWorkflow",
    "CSRDESRSE1Workflow",
    "MultiFrameworkWorkflow",
    # --- Templates (15 + Registry) ---
    "SBTiProgressTemplate",
    "CDPGovernanceTemplate",
    "CDPEmissionsTemplate",
    "TCFDGovernanceTemplate",
    "TCFDStrategyTemplate",
    "TCFDRiskTemplate",
    "TCFDMetricsTemplate",
    "GRI305Template",
    "ISSBS2Template",
    "SECClimateTemplate",
    "CSRDE1Template",
    "InvestorDashboardTemplate",
    "RegulatorDashboardTemplate",
    "CustomerCarbonTemplate",
    "AssuranceEvidenceTemplate",
    "TemplateRegistry",
    # --- Integrations (12) ---
    "PACK021Integration",
    "PACK022Integration",
    "PACK028Integration",
    "PACK029Integration",
    "GLSBTiAppIntegration",
    "GLCDPAppIntegration",
    "GLTCFDAppIntegration",
    "GLGHGAppIntegration",
    "XBRLTaxonomyIntegration",
    "TranslationIntegration",
    "OrchestratorIntegration",
    "HealthCheckIntegration",
]
