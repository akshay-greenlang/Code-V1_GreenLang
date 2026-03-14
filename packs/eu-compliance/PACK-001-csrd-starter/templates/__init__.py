# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Report Template Generators
========================================================

Phase 3 report templates for CSRD compliance reporting. Each template
generates structured output in Markdown, HTML, and JSON formats with
full provenance tracking and ESRS-aligned data models.

Templates:
    - ExecutiveSummaryTemplate: Board-level 2-page CSRD summary
    - ESRSDisclosureTemplate: Full ESRS disclosure narrative (12 standards)
    - MaterialityMatrixTemplate: Double materiality assessment report
    - GHGEmissionsReportTemplate: Comprehensive GHG emissions report
    - AuditorPackageTemplate: External auditor evidence package
    - ComplianceDashboardTemplate: Real-time compliance dashboard data

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_001_csrd_starter.templates.executive_summary import (
    ExecutiveSummaryTemplate,
    ExecutiveSummaryInput,
    ComplianceStatusEntry,
    KeyMetricsDashboard,
    MaterialTopicSummary,
    RegulatoryDeadline,
    RiskHeatmapEntry,
    ActionItem,
)
from packs.eu_compliance.PACK_001_csrd_starter.templates.esrs_disclosure import (
    ESRSDisclosureTemplate,
    ESRSDisclosureInput,
    StandardDisclosure,
    DisclosureRequirement,
    MetricValue,
    CrossReference,
    DataQualityIndicator,
)
from packs.eu_compliance.PACK_001_csrd_starter.templates.materiality_matrix import (
    MaterialityMatrixTemplate,
    MaterialityMatrixInput,
    MaterialTopic,
    ImpactMaterialityScores,
    FinancialMaterialityScores,
    MatrixDataPoint,
    StakeholderEngagement,
)
from packs.eu_compliance.PACK_001_csrd_starter.templates.ghg_emissions_report import (
    GHGEmissionsReportTemplate,
    GHGEmissionsInput,
    Scope1Breakdown,
    Scope2Breakdown,
    Scope3Category,
    IntensityMetric,
    EmissionTrend,
    MethodologyReference,
)
from packs.eu_compliance.PACK_001_csrd_starter.templates.auditor_package import (
    AuditorPackageTemplate,
    AuditorPackageInput,
    CalculationAuditEntry,
    DataLineageRecord,
    SourceDataReference,
    ComplianceChecklistItem,
    DataQualityAssessment,
)
from packs.eu_compliance.PACK_001_csrd_starter.templates.compliance_dashboard import (
    ComplianceDashboardTemplate,
    ComplianceDashboardInput,
    StandardComplianceEntry,
    DataCompletenessCell,
    OutstandingAction,
    ComplianceTrendPoint,
    AlertEntry,
    UpcomingDeadline,
)

__all__ = [
    # Executive Summary
    "ExecutiveSummaryTemplate",
    "ExecutiveSummaryInput",
    "ComplianceStatusEntry",
    "KeyMetricsDashboard",
    "MaterialTopicSummary",
    "RegulatoryDeadline",
    "RiskHeatmapEntry",
    "ActionItem",
    # ESRS Disclosure
    "ESRSDisclosureTemplate",
    "ESRSDisclosureInput",
    "StandardDisclosure",
    "DisclosureRequirement",
    "MetricValue",
    "CrossReference",
    "DataQualityIndicator",
    # Materiality Matrix
    "MaterialityMatrixTemplate",
    "MaterialityMatrixInput",
    "MaterialTopic",
    "ImpactMaterialityScores",
    "FinancialMaterialityScores",
    "MatrixDataPoint",
    "StakeholderEngagement",
    # GHG Emissions
    "GHGEmissionsReportTemplate",
    "GHGEmissionsInput",
    "Scope1Breakdown",
    "Scope2Breakdown",
    "Scope3Category",
    "IntensityMetric",
    "EmissionTrend",
    "MethodologyReference",
    # Auditor Package
    "AuditorPackageTemplate",
    "AuditorPackageInput",
    "CalculationAuditEntry",
    "DataLineageRecord",
    "SourceDataReference",
    "ComplianceChecklistItem",
    "DataQualityAssessment",
    # Compliance Dashboard
    "ComplianceDashboardTemplate",
    "ComplianceDashboardInput",
    "StandardComplianceEntry",
    "DataCompletenessCell",
    "OutstandingAction",
    "ComplianceTrendPoint",
    "AlertEntry",
    "UpcomingDeadline",
]
