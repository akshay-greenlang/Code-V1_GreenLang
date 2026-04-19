# -*- coding: utf-8 -*-
"""
PACK-023: SBTi Alignment Pack
=============================================================================

Comprehensive GreenLang deployment pack for Science Based Targets initiative
(SBTi) alignment lifecycle management. Provides 10 engines covering target
setting (ACA/SDA/FLAG pathway selection, 1.5C/WB2C/2C ambition), 42-criterion
automated validation (28 near-term C1-C28, 14 net-zero NZ-C1 to NZ-C14),
15-category Scope 3 materiality screening with 40% trigger assessment,
SDA intensity convergence for 12 homogeneous sectors, FLAG assessment with
11 commodity categories and 3.03%/yr linear reduction, temperature rating
(SBTi TR v2.0 with 6 aggregation methods), annual progress tracking with
RAG status, base year recalculation with 5% significance threshold,
FI portfolio targets (FINZ V1.0 with 8 asset classes and PCAF scoring),
and 5-dimension submission readiness assessment.

Supports 8 workflows (target setting, validation, Scope 3 assessment, SDA
pathway, FLAG assessment, progress review, FI target setting, full SBTi
lifecycle) and 10 report templates covering target summaries through
submission packages.

Components:
    Engines (10):
        - TargetSettingEngine           (ACA/SDA/FLAG target definition)
        - CriteriaValidationEngine      (42-criterion compliance)
        - Scope3ScreeningEngine         (15-category materiality screening)
        - RecalculationEngine           (Base year recalculation)
        - FLAGAssessmentEngine          (FLAG 11-commodity assessment)
        - TemperatureRatingEngine       (SBTi TR v2.0 scoring)
        - ProgressTrackingEngine        (Annual RAG tracking)
        - SDASectorEngine               (12-sector SDA convergence)
        - FIPortfolioEngine             (FINZ V1.0 portfolio targets)
        - SubmissionReadinessEngine     (5-dimension readiness)

    Workflows (8):
        - TargetSettingWorkflow         (6 phases)
        - ValidationWorkflow            (5 phases)
        - Scope3AssessmentWorkflow      (5 phases)
        - SDAPathwayWorkflow            (6 phases)
        - FLAGWorkflow                  (5 phases)
        - ProgressReviewWorkflow        (6 phases)
        - FITargetWorkflow              (5 phases)
        - FullSBTiLifecycleWorkflow     (10 phases)

    Templates (10):
        - TargetSummaryReportTemplate
        - ValidationReportTemplate
        - Scope3ScreeningReportTemplate
        - SDAPathwayReportTemplate
        - FLAGAssessmentReportTemplate
        - TemperatureRatingReportTemplate
        - ProgressDashboardReportTemplate
        - FIPortfolioReportTemplate
        - SubmissionPackageReportTemplate
        - FrameworkCrosswalkReportTemplate

    Integrations (12):
        - SBTiAlignmentOrchestrator     (10-phase DAG pipeline)
        - SBTiAppBridge                 (GL-SBTi-APP bridge)
        - SBTiMRVBridge                 (30-agent MRV routing)
        - SBTiGHGAppBridge              (GL-GHG-APP bridge)
        - Pack021Bridge                 (PACK-021 optional bridge)
        - Pack022Bridge                 (PACK-022 optional bridge)
        - SBTiDecarbBridge              (21 DECARB agent bridge)
        - SBTiDataBridge                (20 DATA agent bridge)
        - SBTiReportingBridge           (Cross-framework reporting)
        - SBTiOffsetBridge              (Carbon credit management)
        - SBTiHealthCheck               (20-category health check)
        - SBTiAlignmentSetupWizard      (6-step setup wizard)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

__version__: str = "1.0.0"
__pack__: str = "PACK-023"
__pack_name__: str = "SBTi Alignment Pack"
__category__: str = "net-zero"

__all__: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__category__",
]
