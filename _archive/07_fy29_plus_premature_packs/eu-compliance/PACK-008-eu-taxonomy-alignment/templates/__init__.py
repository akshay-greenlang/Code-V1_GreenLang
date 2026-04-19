# -*- coding: utf-8 -*-
"""PACK-008 EU Taxonomy Alignment Pack report templates.

This package provides 10 report and disclosure templates for EU Taxonomy
alignment assessment, covering eligibility screening through Article 8
disclosure generation and EBA Pillar 3 GAR reporting.

Templates:
    1. EligibilityMatrixReportTemplate - Activity-level eligibility per objective
    2. AlignmentSummaryReportTemplate - Portfolio-level alignment results
    3. Article8DisclosureTemplate - Mandatory disclosure tables (Turnover/CapEx/OpEx)
    4. EBAPillar3GARReportTemplate - EBA Pillar 3 Templates 6-10
    5. KPIDashboardTemplate - Turnover/CapEx/OpEx alignment dashboards
    6. GapAnalysisReportTemplate - Gap inventory with remediation roadmap
    7. TSCComplianceReportTemplate - Technical screening criteria results
    8. DNSHAssessmentReportTemplate - 6-objective DNSH matrix
    9. ExecutiveSummaryTemplate - Board-level overview
    10. DetailedAssessmentReportTemplate - Full audit trail with provenance
"""

from .eligibility_matrix_report import EligibilityMatrixReportTemplate
from .alignment_summary_report import AlignmentSummaryReportTemplate
from .article8_disclosure_template import Article8DisclosureTemplate
from .eba_pillar3_gar_report import EBAPillar3GARReportTemplate
from .kpi_dashboard import KPIDashboardTemplate
from .gap_analysis_report import GapAnalysisReportTemplate
from .tsc_compliance_report import TSCComplianceReportTemplate
from .dnsh_assessment_report import DNSHAssessmentReportTemplate
from .executive_summary import ExecutiveSummaryTemplate
from .detailed_assessment_report import DetailedAssessmentReportTemplate

__all__ = [
    "EligibilityMatrixReportTemplate",
    "AlignmentSummaryReportTemplate",
    "Article8DisclosureTemplate",
    "EBAPillar3GARReportTemplate",
    "KPIDashboardTemplate",
    "GapAnalysisReportTemplate",
    "TSCComplianceReportTemplate",
    "DNSHAssessmentReportTemplate",
    "ExecutiveSummaryTemplate",
    "DetailedAssessmentReportTemplate",
]
