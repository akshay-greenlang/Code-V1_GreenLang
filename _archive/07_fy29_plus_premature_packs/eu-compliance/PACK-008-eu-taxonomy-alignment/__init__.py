# -*- coding: utf-8 -*-
"""
PACK-008: EU Taxonomy Alignment Pack
=============================================================================

Comprehensive GreenLang deployment pack for EU Taxonomy alignment
assessment per EU Taxonomy Regulation (EU) 2020/852 and associated
Delegated Acts. Provides 10 engines covering activity eligibility
screening across 6 environmental objectives, substantial contribution
threshold assessment, Do No Significant Harm (DNSH) 6-objective
evaluation, OECD/UNGP/ILO minimum safeguards checking,
Turnover/CapEx/OpEx KPI calculation, EBA Pillar 3 Green Asset Ratio
computation, Technical Screening Criteria evaluation against
Delegated Acts, Article 10(2) transition activity assessment,
Article 16 enabling activity assessment, and Article 8 mandatory
disclosure generation with XBRL tagging.

Components:
    Engines (10):
        - TaxonomyEligibilityEngine
        - SubstantialContributionEngine
        - DNSHAssessmentEngine
        - MinimumSafeguardsEngine
        - KPICalculationEngine
        - GreenAssetRatioEngine
        - TechnicalScreeningCriteriaEngine
        - TransitionActivityEngine
        - EnablingActivityEngine
        - TaxonomyReportingEngine

    Workflows (10):
        - EligibilityScreeningWorkflow
        - AlignmentAssessmentWorkflow
        - KPICalculationWorkflow
        - GARCalculationWorkflow
        - Article8DisclosureWorkflow
        - GapAnalysisWorkflow
        - CapExPlanWorkflow
        - RegulatoryUpdateWorkflow
        - CrossFrameworkAlignmentWorkflow
        - AnnualTaxonomyReviewWorkflow

    Templates (10):
        - EligibilityMatrixReportTemplate
        - AlignmentSummaryReportTemplate
        - Article8DisclosureTemplate
        - EBAPillar3GARReportTemplate
        - KPIDashboardTemplate
        - GapAnalysisReportTemplate
        - TSCComplianceReportTemplate
        - DNSHAssessmentReportTemplate
        - ExecutiveSummaryTemplate
        - DetailedAssessmentReportTemplate

    Integrations (12):
        - TaxonomyPackOrchestrator
        - TaxonomyAppBridge
        - MRVTaxonomyBridge
        - CSRDCrossFrameworkBridge
        - FinancialDataBridge
        - ActivityRegistryBridge
        - GARDataBridge
        - EvidenceManagementBridge
        - RegulatoryTrackingBridge
        - DataQualityBridge
        - TaxonomyHealthCheck
        - TaxonomySetupWizard

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__: str = "1.0.0"
__pack__: str = "PACK-008"
__pack_name__: str = "EU Taxonomy Alignment Pack"
__category__: str = "eu-compliance"
__author__: str = "GreenLang Platform Team"
