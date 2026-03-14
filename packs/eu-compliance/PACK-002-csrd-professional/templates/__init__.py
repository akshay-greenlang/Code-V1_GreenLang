# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Report Template Generators
==============================================================

Phase 3 report templates for CSRD professional compliance reporting.
Each template generates structured output in Markdown, HTML, and JSON
formats with full provenance tracking and ESRS-aligned data models.

Templates:
    - ConsolidatedReportTemplate: Multi-entity consolidated ESRS report
    - CrossFrameworkReportTemplate: Cross-framework alignment map
    - ScenarioAnalysisReportTemplate: Climate scenario analysis results
    - InvestorESGReportTemplate: Investor-focused ESG report
    - BoardGovernancePackTemplate: Board-level sustainability governance pack
    - RegulatoryFilingPackageTemplate: Regulatory filing package
    - BenchmarkingDashboardTemplate: Peer comparison dashboard
    - StakeholderReportTemplate: Stakeholder engagement documentation
    - DataGovernanceReportTemplate: Data governance status report
    - ProfessionalDashboardTemplate: Enhanced real-time compliance dashboard

Author: GreenLang Team
Version: 2.0.0
"""

from packs.eu_compliance.PACK_002_csrd_professional.templates.consolidated_report import (
    ConsolidatedReportTemplate,
    ConsolidatedReportInput,
    EntitySummary,
    ConsolidatedEmissions,
    StandardDisclosure,
    EliminationEntry,
    ReconciliationEntry,
    ConsolidationApproach,
    CoverageStatus,
    ReconciliationStatus,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.cross_framework_report import (
    CrossFrameworkReportTemplate,
    CrossFrameworkReportInput,
    FrameworkAlignment,
    GapEntry,
    CDPScoringResult,
    SBTiResult,
    TaxonomyResult,
    AlignmentStatus,
    GapPriority,
    CDPScoreGrade,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.scenario_analysis_report import (
    ScenarioAnalysisReportTemplate,
    ScenarioAnalysisReportInput,
    ScenarioSummary,
    PhysicalRiskEntry,
    TransitionRiskEntry,
    FinancialImpactEntry,
    ResilienceAssessment,
    MACCEntry,
    ScenarioType,
    PhysicalRiskType,
    TransitionRiskDriver,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.investor_esg_report import (
    InvestorESGReportTemplate,
    InvestorESGReportInput,
    ESGScores,
    RatingPrediction,
    PeerBenchmark,
    SBTiStatus,
    TaxonomyKPIs,
    ClimateRiskSummary,
    TargetProgress as InvestorTargetProgress,
    RatingAgency,
    TargetTrackingStatus,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.board_governance_pack import (
    BoardGovernancePackTemplate,
    BoardGovernancePackInput,
    GovernanceStructure,
    KPIEntry,
    RiskEntry,
    ComplianceStatus as BoardComplianceStatus,
    TargetProgress as BoardTargetProgress,
    DecisionItem,
    KPIStatus,
    KPITrend,
    RiskCategory,
    DecisionUrgency,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.regulatory_filing_package import (
    RegulatoryFilingPackageTemplate,
    RegulatoryFilingInput,
    JurisdictionFiling,
    ESEFPackageStatus,
    FilingRecord,
    SignatureStatus,
    FilingStatus,
    FormatRequired,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.benchmarking_dashboard import (
    BenchmarkingDashboardTemplate,
    BenchmarkingDashboardInput,
    PeerComparison,
    TrendMetric,
    ImprovementPriority,
    SectorLeader,
    Quartile,
    EffortLevel,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.stakeholder_report import (
    StakeholderReportTemplate,
    StakeholderReportInput,
    StakeholderSummary,
    SalienceAnalysis,
    EngagementActivitySummary,
    MaterialityInfluence,
    ParticipationMetrics,
    EvidenceSummary,
    StakeholderCategory,
    EngagementType,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.data_governance_report import (
    DataGovernanceReportTemplate,
    DataGovernanceReportInput,
    ClassificationSummary,
    RetentionCompliance,
    GDPRStatus,
    QualitySLA,
    SLATarget,
    AuditFinding,
    ClassificationLevel,
    FindingSeverity,
    RemediationStatus,
)
from packs.eu_compliance.PACK_002_csrd_professional.templates.professional_dashboard import (
    ProfessionalDashboardTemplate,
    ProfessionalDashboardInput,
    StandardCompliance,
    QualityGateStatus,
    ApprovalPipelineStatus,
    RegulatoryAlert,
    BenchmarkPosition,
    EntityComplianceStatus,
    SLOStatus,
    DeadlineEntry,
    AlertSeverity,
    SLOStatusLevel,
    DeadlineStatus,
)

__all__ = [
    # Consolidated Report
    "ConsolidatedReportTemplate",
    "ConsolidatedReportInput",
    "EntitySummary",
    "ConsolidatedEmissions",
    "StandardDisclosure",
    "EliminationEntry",
    "ReconciliationEntry",
    "ConsolidationApproach",
    "CoverageStatus",
    "ReconciliationStatus",
    # Cross-Framework Report
    "CrossFrameworkReportTemplate",
    "CrossFrameworkReportInput",
    "FrameworkAlignment",
    "GapEntry",
    "CDPScoringResult",
    "SBTiResult",
    "TaxonomyResult",
    "AlignmentStatus",
    "GapPriority",
    "CDPScoreGrade",
    # Scenario Analysis Report
    "ScenarioAnalysisReportTemplate",
    "ScenarioAnalysisReportInput",
    "ScenarioSummary",
    "PhysicalRiskEntry",
    "TransitionRiskEntry",
    "FinancialImpactEntry",
    "ResilienceAssessment",
    "MACCEntry",
    "ScenarioType",
    "PhysicalRiskType",
    "TransitionRiskDriver",
    # Investor ESG Report
    "InvestorESGReportTemplate",
    "InvestorESGReportInput",
    "ESGScores",
    "RatingPrediction",
    "PeerBenchmark",
    "SBTiStatus",
    "TaxonomyKPIs",
    "ClimateRiskSummary",
    "InvestorTargetProgress",
    "RatingAgency",
    "TargetTrackingStatus",
    # Board Governance Pack
    "BoardGovernancePackTemplate",
    "BoardGovernancePackInput",
    "GovernanceStructure",
    "KPIEntry",
    "RiskEntry",
    "BoardComplianceStatus",
    "BoardTargetProgress",
    "DecisionItem",
    "KPIStatus",
    "KPITrend",
    "RiskCategory",
    "DecisionUrgency",
    # Regulatory Filing Package
    "RegulatoryFilingPackageTemplate",
    "RegulatoryFilingInput",
    "JurisdictionFiling",
    "ESEFPackageStatus",
    "FilingRecord",
    "SignatureStatus",
    "FilingStatus",
    "FormatRequired",
    # Benchmarking Dashboard
    "BenchmarkingDashboardTemplate",
    "BenchmarkingDashboardInput",
    "PeerComparison",
    "TrendMetric",
    "ImprovementPriority",
    "SectorLeader",
    "Quartile",
    "EffortLevel",
    # Stakeholder Report
    "StakeholderReportTemplate",
    "StakeholderReportInput",
    "StakeholderSummary",
    "SalienceAnalysis",
    "EngagementActivitySummary",
    "MaterialityInfluence",
    "ParticipationMetrics",
    "EvidenceSummary",
    "StakeholderCategory",
    "EngagementType",
    # Data Governance Report
    "DataGovernanceReportTemplate",
    "DataGovernanceReportInput",
    "ClassificationSummary",
    "RetentionCompliance",
    "GDPRStatus",
    "QualitySLA",
    "SLATarget",
    "AuditFinding",
    "ClassificationLevel",
    "FindingSeverity",
    "RemediationStatus",
    # Professional Dashboard
    "ProfessionalDashboardTemplate",
    "ProfessionalDashboardInput",
    "StandardCompliance",
    "QualityGateStatus",
    "ApprovalPipelineStatus",
    "RegulatoryAlert",
    "BenchmarkPosition",
    "EntityComplianceStatus",
    "SLOStatus",
    "DeadlineEntry",
    "AlertSeverity",
    "SLOStatusLevel",
    "DeadlineStatus",
]
