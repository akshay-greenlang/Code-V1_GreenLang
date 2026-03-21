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

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template 1: Consolidated Report
# ---------------------------------------------------------------------------
_TEMPLATE_1_SYMBOLS: list[str] = [
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
]

try:
    from .consolidated_report import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 1 (ConsolidatedReportTemplate) not available: %s", e)
    _TEMPLATE_1_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 2: Cross-Framework Report
# ---------------------------------------------------------------------------
_TEMPLATE_2_SYMBOLS: list[str] = [
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
]

try:
    from .cross_framework_report import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 2 (CrossFrameworkReportTemplate) not available: %s", e)
    _TEMPLATE_2_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 3: Scenario Analysis Report
# ---------------------------------------------------------------------------
_TEMPLATE_3_SYMBOLS: list[str] = [
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
]

try:
    from .scenario_analysis_report import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 3 (ScenarioAnalysisReportTemplate) not available: %s", e)
    _TEMPLATE_3_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 4: Investor ESG Report
# ---------------------------------------------------------------------------
_TEMPLATE_4_SYMBOLS: list[str] = [
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
]

try:
    from .investor_esg_report import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 4 (InvestorESGReportTemplate) not available: %s", e)
    _TEMPLATE_4_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 5: Board Governance Pack
# ---------------------------------------------------------------------------
_TEMPLATE_5_SYMBOLS: list[str] = [
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
]

try:
    from .board_governance_pack import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 5 (BoardGovernancePackTemplate) not available: %s", e)
    _TEMPLATE_5_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 6: Regulatory Filing Package
# ---------------------------------------------------------------------------
_TEMPLATE_6_SYMBOLS: list[str] = [
    "RegulatoryFilingPackageTemplate",
    "RegulatoryFilingInput",
    "JurisdictionFiling",
    "ESEFPackageStatus",
    "FilingRecord",
    "SignatureStatus",
    "FilingStatus",
    "FormatRequired",
]

try:
    from .regulatory_filing_package import (  # noqa: F401
        RegulatoryFilingPackageTemplate,
        RegulatoryFilingInput,
        JurisdictionFiling,
        ESEFPackageStatus,
        FilingRecord,
        SignatureStatus,
        FilingStatus,
        FormatRequired,
    )
except ImportError as e:
    logger.debug("Template 6 (RegulatoryFilingPackageTemplate) not available: %s", e)
    _TEMPLATE_6_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 7: Benchmarking Dashboard
# ---------------------------------------------------------------------------
_TEMPLATE_7_SYMBOLS: list[str] = [
    "BenchmarkingDashboardTemplate",
    "BenchmarkingDashboardInput",
    "PeerComparison",
    "TrendMetric",
    "ImprovementPriority",
    "SectorLeader",
    "Quartile",
    "EffortLevel",
]

try:
    from .benchmarking_dashboard import (  # noqa: F401
        BenchmarkingDashboardTemplate,
        BenchmarkingDashboardInput,
        PeerComparison,
        TrendMetric,
        ImprovementPriority,
        SectorLeader,
        Quartile,
        EffortLevel,
    )
except ImportError as e:
    logger.debug("Template 7 (BenchmarkingDashboardTemplate) not available: %s", e)
    _TEMPLATE_7_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 8: Stakeholder Report
# ---------------------------------------------------------------------------
_TEMPLATE_8_SYMBOLS: list[str] = [
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
]

try:
    from .stakeholder_report import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 8 (StakeholderReportTemplate) not available: %s", e)
    _TEMPLATE_8_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 9: Data Governance Report
# ---------------------------------------------------------------------------
_TEMPLATE_9_SYMBOLS: list[str] = [
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
]

try:
    from .data_governance_report import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 9 (DataGovernanceReportTemplate) not available: %s", e)
    _TEMPLATE_9_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 10: Professional Dashboard
# ---------------------------------------------------------------------------
_TEMPLATE_10_SYMBOLS: list[str] = [
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

try:
    from .professional_dashboard import (  # noqa: F401
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
except ImportError as e:
    logger.debug("Template 10 (ProfessionalDashboardTemplate) not available: %s", e)
    _TEMPLATE_10_SYMBOLS = []


# ---------------------------------------------------------------------------
# Template catalog and registry
# ---------------------------------------------------------------------------
TEMPLATE_CATALOG: list[dict[str, Any]] = []

if _TEMPLATE_1_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "consolidated_report",
        "class": ConsolidatedReportTemplate,
        "description": "Multi-entity consolidated ESRS report.",
        "category": "report",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_2_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "cross_framework_report",
        "class": CrossFrameworkReportTemplate,
        "description": "Cross-framework alignment map.",
        "category": "report",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_3_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "scenario_analysis_report",
        "class": ScenarioAnalysisReportTemplate,
        "description": "Climate scenario analysis results.",
        "category": "analytics",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_4_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "investor_esg_report",
        "class": InvestorESGReportTemplate,
        "description": "Investor-focused ESG report.",
        "category": "report",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_5_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "board_governance_pack",
        "class": BoardGovernancePackTemplate,
        "description": "Board-level sustainability governance pack.",
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_6_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "regulatory_filing_package",
        "class": RegulatoryFilingPackageTemplate,
        "description": "Regulatory filing package.",
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_7_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "benchmarking_dashboard",
        "class": BenchmarkingDashboardTemplate,
        "description": "Peer comparison dashboard.",
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_8_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "stakeholder_report",
        "class": StakeholderReportTemplate,
        "description": "Stakeholder engagement documentation.",
        "category": "report",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_9_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "data_governance_report",
        "class": DataGovernanceReportTemplate,
        "description": "Data governance status report.",
        "category": "report",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })
if _TEMPLATE_10_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "professional_dashboard",
        "class": ProfessionalDashboardTemplate,
        "description": "Enhanced real-time compliance dashboard.",
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "2.0.0",
    })


class TemplateRegistry:
    """
    Registry for PACK-002 CSRD Professional Pack report templates.

    Provides a centralized way to discover, instantiate, and manage
    all 10 professional report templates. Templates can be listed,
    retrieved by name, and instantiated with optional configuration.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> template = registry.get("consolidated_report")
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}

        for defn in TEMPLATE_CATALOG:
            self._templates[defn["name"]] = defn

        logger.info(
            "PACK-002 TemplateRegistry initialized with %d templates",
            len(self._templates),
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with metadata."""
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
        ]

    def list_template_names(self) -> List[str]:
        """List all available template names."""
        return [defn["name"] for defn in TEMPLATE_CATALOG]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get a template instance by name.

        Args:
            name: Template name (e.g., 'consolidated_report').
            config: Optional configuration overrides.

        Returns:
            Template instance.

        Raises:
            KeyError: If template name is not found.
        """
        if name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. Available: {available}"
            )

        if config is not None or name not in self._instances:
            template_class = self._templates[name]["class"]
            instance = template_class(config=config)
            if config is None:
                self._instances[name] = instance
            return instance

        return self._instances[name]

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a specific template.

        Args:
            name: Template name.

        Returns:
            Template info dict.

        Raises:
            KeyError: If template name is not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")

        defn = self._templates[name]
        return {
            "name": defn["name"],
            "description": defn["description"],
            "category": defn["category"],
            "formats": defn["formats"],
            "version": defn["version"],
            "class_name": defn["class"].__name__,
        }

    def has_template(self, name: str) -> bool:
        """Check if a template exists by name."""
        return name in self._templates

    @property
    def template_count(self) -> int:
        """Return the number of registered templates."""
        return len(self._templates)

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return (
            f"TemplateRegistry(templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )


__all__: list[str] = [
    # Template Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    # Template classes and models
    *_TEMPLATE_1_SYMBOLS,
    *_TEMPLATE_2_SYMBOLS,
    *_TEMPLATE_3_SYMBOLS,
    *_TEMPLATE_4_SYMBOLS,
    *_TEMPLATE_5_SYMBOLS,
    *_TEMPLATE_6_SYMBOLS,
    *_TEMPLATE_7_SYMBOLS,
    *_TEMPLATE_8_SYMBOLS,
    *_TEMPLATE_9_SYMBOLS,
    *_TEMPLATE_10_SYMBOLS,
]
