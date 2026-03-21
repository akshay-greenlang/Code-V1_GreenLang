# -*- coding: utf-8 -*-
"""
PACK-029: Interim Targets Pack
=============================================================================

Comprehensive GreenLang deployment pack for interim target management,
progress tracking, and corrective action planning along net-zero pathways.
Provides 10 engines covering SBTi-aligned interim target setting (near-term
2030, long-term 2040/2050, annual/quarterly milestones), linear/non-linear
annual pathway decomposition, multi-dimensional progress tracking with
leading/lagging KPIs, statistical variance analysis with root cause
attribution, AI-free trend extrapolation using regression/ARIMA/exponential
smoothing, corrective action planning with MACC-ranked interventions,
milestone validation against SBTi/CDP/TCFD criteria, initiative scheduling
with dependency management and resource allocation, carbon budget allocation
across scopes/BUs/categories, and multi-framework progress reporting with
assurance evidence packaging.

Supports quarterly monitoring cycles with automatic variance alerts,
trajectory recalibration, and stakeholder reporting across SBTi/CDP/TCFD
frameworks.

Components:
    Engines (10):
        - InterimTargetEngine              (SBTi 1.5C/WB2C target validation)
        - AnnualPathwayEngine              (Linear/non-linear decomposition)
        - ProgressTrackerEngine            (Multi-KPI progress tracking)
        - VarianceAnalysisEngine           (Root cause attribution)
        - TrendExtrapolationEngine         (Regression/ARIMA forecasting)
        - CorrectiveActionEngine           (MACC-ranked interventions)
        - MilestoneValidationEngine        (SBTi/CDP/TCFD criteria check)
        - InitiativeSchedulerEngine        (Dependency-aware scheduling)
        - BudgetAllocationEngine           (Carbon budget by scope/BU)
        - ReportingEngine                  (Multi-framework reporting)

    Workflows (7):
        - InterimTargetSettingWorkflow           (5 phases)
        - AnnualProgressReviewWorkflow           (4 phases)
        - QuarterlyMonitoringWorkflow            (4 phases)
        - VarianceInvestigationWorkflow          (4 phases)
        - CorrectiveActionPlanningWorkflow       (5 phases)
        - AnnualReportingWorkflow                (5 phases)
        - TargetRecalibrationWorkflow            (4 phases)

    Templates (10):
        - InterimTargetsSummaryTemplate
        - AnnualProgressReportTemplate
        - VarianceAnalysisReportTemplate
        - CorrectiveActionPlanTemplate
        - QuarterlyDashboardTemplate
        - CDPDisclosureTemplate
        - TCFDMetricsReportTemplate
        - AssuranceEvidencePackageTemplate
        - ExecutiveSummaryTemplate
        - PublicDisclosureTemplate

    Integrations (10):
        - PACK021Bridge                    (Baseline/target import)
        - PACK028Bridge                    (Sector pathway import)
        - MRVBridge                        (30 MRV agents)
        - SBTiBridge                       (SBTi portal sync)
        - CDPBridge                        (CDP questionnaire)
        - TCFDBridge                       (TCFD disclosure)
        - InitiativeTrackerBridge          (Project management)
        - BudgetSystemBridge               (Financial systems)
        - AlertingBridge                   (Notification dispatch)
        - AssurancePortalBridge            (Auditor evidence)

    Presets (7):
        - corporate_standard       (Large Corporate, near-term 2030)
        - financial_institution    (Banks/Insurance, PCAF-aligned)
        - heavy_emitter            (Heavy Industry, SDA pathway)
        - sme_simplified           (SMEs, simplified tracking)
        - city_municipality        (Cities, C40/GPC-aligned)
        - quarterly_monitor        (Quarterly cycle focus)
        - annual_reporter          (Annual reporting focus)

Agent Dependencies:
    - 30 AGENT-MRV agents (Scope 1/2/3 emissions quantification)
    - 20 AGENT-DATA agents (data intake and quality management)
    - 10 AGENT-FOUND agents (platform foundation services)

Regulatory Framework:
    Primary:
        - SBTi Corporate Net-Zero Standard v1.2 (2024)
        - SBTi Near-Term Target Setting Guidance (2024)
        - CDP Climate Change Questionnaire (2024)
        - TCFD Recommendations (2017, updated 2023)
    Secondary:
        - IPCC AR6 WG3 (2022) -- 1.5C pathway budgets
        - GHG Protocol Corporate/Scope 3 Standards
        - ISSB IFRS S2 (2023) -- Climate-related disclosures
        - ISAE 3410 -- Assurance on GHG statements
        - Paris Agreement (2015) -- Temperature targets

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-029"
__pack_name__ = "Interim Targets Pack"
__author__ = "GreenLang Platform Team"
__category__: str = "net-zero"

# ---------------------------------------------------------------------------
# Engines (10)
# ---------------------------------------------------------------------------
from .engines import (
    InterimTargetEngine,
    AnnualPathwayEngine,
    ProgressTrackerEngine,
    VarianceAnalysisEngine,
    TrendExtrapolationEngine,
    CorrectiveActionEngine,
    MilestoneValidationEngine,
    InitiativeSchedulerEngine,
    BudgetAllocationEngine,
    ReportingEngine,
)

# ---------------------------------------------------------------------------
# Workflows (7)
# ---------------------------------------------------------------------------
from .workflows import (
    InterimTargetSettingWorkflow,
    AnnualProgressReviewWorkflow,
    QuarterlyMonitoringWorkflow,
    VarianceInvestigationWorkflow,
    CorrectiveActionPlanningWorkflow,
    AnnualReportingWorkflow,
    TargetRecalibrationWorkflow,
)

# ---------------------------------------------------------------------------
# Templates (10 + Registry)
# ---------------------------------------------------------------------------
from .templates import (
    InterimTargetsSummaryTemplate,
    AnnualProgressReportTemplate,
    VarianceAnalysisReportTemplate,
    CorrectiveActionPlanTemplate,
    QuarterlyDashboardTemplate,
    CDPDisclosureTemplate,
    TCFDMetricsReportTemplate,
    AssuranceEvidencePackageTemplate,
    ExecutiveSummaryTemplate,
    PublicDisclosureTemplate,
    TemplateRegistry,
)

# ---------------------------------------------------------------------------
# Integrations (10)
# ---------------------------------------------------------------------------
from .integrations import (
    PACK021Bridge,
    PACK028Bridge,
    MRVBridge,
    SBTiBridge,
    CDPBridge,
    TCFDBridge,
    InitiativeTrackerBridge,
    BudgetSystemBridge,
    AlertingBridge,
    AssurancePortalBridge,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__author__",
    # --- Engines (10) ---
    "InterimTargetEngine",
    "AnnualPathwayEngine",
    "ProgressTrackerEngine",
    "VarianceAnalysisEngine",
    "TrendExtrapolationEngine",
    "CorrectiveActionEngine",
    "MilestoneValidationEngine",
    "InitiativeSchedulerEngine",
    "BudgetAllocationEngine",
    "ReportingEngine",
    # --- Workflows (7) ---
    "InterimTargetSettingWorkflow",
    "AnnualProgressReviewWorkflow",
    "QuarterlyMonitoringWorkflow",
    "VarianceInvestigationWorkflow",
    "CorrectiveActionPlanningWorkflow",
    "AnnualReportingWorkflow",
    "TargetRecalibrationWorkflow",
    # --- Templates (10 + Registry) ---
    "InterimTargetsSummaryTemplate",
    "AnnualProgressReportTemplate",
    "VarianceAnalysisReportTemplate",
    "CorrectiveActionPlanTemplate",
    "QuarterlyDashboardTemplate",
    "CDPDisclosureTemplate",
    "TCFDMetricsReportTemplate",
    "AssuranceEvidencePackageTemplate",
    "ExecutiveSummaryTemplate",
    "PublicDisclosureTemplate",
    "TemplateRegistry",
    # --- Integrations (10) ---
    "PACK021Bridge",
    "PACK028Bridge",
    "MRVBridge",
    "SBTiBridge",
    "CDPBridge",
    "TCFDBridge",
    "InitiativeTrackerBridge",
    "BudgetSystemBridge",
    "AlertingBridge",
    "AssurancePortalBridge",
]
