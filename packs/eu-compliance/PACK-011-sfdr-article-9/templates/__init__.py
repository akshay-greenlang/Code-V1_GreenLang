"""
SFDR Article 9 Pack Templates - PACK-011

This module exports all 8 SFDR Article 9 report template classes and provides
a TemplateRegistry for programmatic template discovery, rendering, and
instantiation.

Templates:
    - AnnexIIIPrecontractualTemplate: SFDR RTS Annex III pre-contractual disclosure
    - AnnexVPeriodicTemplate: SFDR RTS Annex V periodic reporting
    - ImpactReportTemplate: Sustainable impact measurement report
    - BenchmarkMethodologyTemplate: CTB/PAB benchmark methodology report
    - SustainableDashboardTemplate: Sustainable investment dashboard
    - PAIMandatoryReportTemplate: All 18 mandatory PAI indicators report
    - CarbonTrajectoryReportTemplate: Carbon intensity trajectory to 2050
    - AuditTrailReportTemplate: Data provenance and audit trail report

Supported Formats:
    - Markdown (GitHub-flavored tables)
    - HTML (self-contained with inline CSS)
    - JSON (structured, machine-readable)

Example:
    >>> from packs.eu_compliance.PACK_011_sfdr_article_9.templates import (
    ...     AnnexIIIPrecontractualTemplate,
    ...     TemplateRegistry,
    ... )
    >>> template = AnnexIIIPrecontractualTemplate()
    >>> markdown = template.render_markdown(data)

    >>> registry = TemplateRegistry()
    >>> names = registry.list_templates()
    >>> template = registry.get_template("annex_iii_precontractual")
    >>> html = registry.render("annex_iii_precontractual", data, "html")
"""

from typing import Any, Dict, List, Optional, Type, Union

from .annex_iii_precontractual import AnnexIIIPrecontractualTemplate
from .annex_v_periodic import AnnexVPeriodicTemplate
from .impact_report import ImpactReportTemplate
from .benchmark_methodology import BenchmarkMethodologyTemplate
from .sustainable_dashboard import SustainableDashboardTemplate
from .pai_mandatory_report import PAIMandatoryReportTemplate
from .carbon_trajectory_report import CarbonTrajectoryReportTemplate
from .audit_trail_report import AuditTrailReportTemplate

# Re-export key data models for convenience
from .annex_iii_precontractual import (
    AnnexIIIPrecontractualData,
    Article9ProductInfo,
    SustainableObjective,
    Article9Strategy,
    Article9Proportions,
    Article9TaxonomyAlignment,
    Article9DNSHApproach,
    Article9DataSources,
    Article9Benchmark,
)
from .impact_report import (
    ImpactReportData,
    ImpactFundInfo,
    TheoryOfChange,
    EnvironmentalKPI,
    SocialKPI,
    SDGContribution,
    YoYComparison,
    AdditionalityAssessment,
)
from .benchmark_methodology import (
    BenchmarkMethodologyData,
    BenchmarkFundInfo,
    BenchmarkSelection,
    CarbonIntensityComparison,
    DecarbonizationTrajectory,
    ExclusionCompliance,
    TrackingErrorAnalysis,
)
from .sustainable_dashboard import (
    SustainableDashboardData,
    DashboardFundInfo,
    SustainableMetrics,
    DNSHMetrics,
    GovernanceMetrics,
    TaxonomyMetrics,
    PAISummaryMetrics,
    ImpactHighlights,
    DowngradeRisk,
    CarbonTrajectoryDashboard,
)
from .pai_mandatory_report import (
    PAIMandatoryReportData,
    PAIEntityInfo,
    MandatoryPAIIndicator,
    AdditionalPAIIndicator,
    PAIIntegrationDescription,
    PAIEngagement,
    PAIDataQuality,
)
from .carbon_trajectory_report import (
    CarbonTrajectoryReportData,
    TrajectoryFundInfo,
    IntensityTrajectory,
    ParisAlignmentAssessment,
    SBTCoverage,
    ImpliedTemperatureRise,
    CarbonBudget,
    NetZeroProgress,
)
from .audit_trail_report import (
    AuditTrailReportData,
    AuditReportInfo,
    ProvenanceRecord,
    DataSourceEntry,
    MethodologyReference,
    DataQualityFlag,
    VersionHistoryEntry,
    SignOffRecord,
)


# Type alias for any template class in this pack
TemplateClass = Union[
    Type[AnnexIIIPrecontractualTemplate],
    Type[AnnexVPeriodicTemplate],
    Type[ImpactReportTemplate],
    Type[BenchmarkMethodologyTemplate],
    Type[SustainableDashboardTemplate],
    Type[PAIMandatoryReportTemplate],
    Type[CarbonTrajectoryReportTemplate],
    Type[AuditTrailReportTemplate],
]

# Supported output formats
SUPPORTED_FORMATS = ("markdown", "html", "json")

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "annex_iii_precontractual": {
        "class": AnnexIIIPrecontractualTemplate,
        "display_name": "Pre-contractual Disclosure (Annex III)",
        "description": (
            "SFDR RTS Annex III pre-contractual disclosure template for "
            "Article 9 products. Covers 10 sections including sustainable "
            "objective, investment strategy, proportions (near 100%), "
            "taxonomy alignment, monitoring, DNSH, data sources, limitations, "
            "due diligence, and designated benchmark."
        ),
        "template_class": AnnexIIIPrecontractualTemplate,
        "category": "regulatory",
        "scope": "pre-contractual",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Sustainable Investment Objective",
            "Investment Strategy",
            "Proportion of Investments",
            "EU Taxonomy Alignment",
            "Monitoring",
            "DNSH Approach",
            "Data Sources",
            "Limitations",
            "Due Diligence",
            "Engagement and Benchmark",
        ],
    },
    "annex_v_periodic": {
        "class": AnnexVPeriodicTemplate,
        "display_name": "Periodic Report (Annex V)",
        "description": (
            "SFDR RTS Annex V periodic reporting template for Article 9 "
            "products. Reports on how the sustainable investment objective "
            "was attained during the reference period, including top "
            "investments, proportion breakdowns, PAI indicators, actions "
            "taken, and benchmark comparison."
        ),
        "template_class": AnnexVPeriodicTemplate,
        "category": "regulatory",
        "scope": "periodic",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Product Identification",
            "Sustainable Objective Attainment",
            "Top Investments",
            "Proportion of Sustainability-Related Investments",
            "PAI Indicators",
            "Actions Taken",
            "Comparison to Previous Period",
            "Benchmark Comparison",
        ],
    },
    "impact_report": {
        "class": ImpactReportTemplate,
        "display_name": "Impact Report",
        "description": (
            "Sustainable impact measurement report for Article 9 products. "
            "Covers Theory of Change framework, 15 environmental KPIs, "
            "12 social KPIs, SDG contribution mapping, year-over-year "
            "comparison, and additionality assessment."
        ),
        "template_class": ImpactReportTemplate,
        "category": "analytics",
        "scope": "impact",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Theory of Change",
            "Environmental KPIs (E01-E15)",
            "Social KPIs (S01-S12)",
            "SDG Contribution Mapping",
            "Year-over-Year Comparison",
            "Additionality Assessment",
        ],
    },
    "benchmark_methodology": {
        "class": BenchmarkMethodologyTemplate,
        "display_name": "Benchmark Methodology",
        "description": (
            "EU Climate Benchmark methodology report for Article 9 products. "
            "Covers CTB/PAB selection rationale, carbon intensity comparison, "
            "decarbonization trajectory, exclusion compliance, and tracking "
            "error analysis."
        ),
        "template_class": BenchmarkMethodologyTemplate,
        "category": "analytics",
        "scope": "benchmark",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Benchmark Selection Rationale",
            "Carbon Intensity Comparison",
            "Decarbonization Trajectory",
            "Exclusion Compliance",
            "Tracking Error & Performance",
        ],
    },
    "sustainable_dashboard": {
        "class": SustainableDashboardTemplate,
        "display_name": "Sustainable Investment Dashboard",
        "description": (
            "Interactive-style sustainable investment dashboard for Article 9 "
            "products. Features ASCII gauges for sustainable proportion, DNSH "
            "rate, governance pass rate, taxonomy alignment, PAI summary, "
            "impact highlights, downgrade risk, and carbon trajectory."
        ),
        "template_class": SustainableDashboardTemplate,
        "category": "analytics",
        "scope": "dashboard",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Key Compliance Gauges",
            "Sustainable Allocation",
            "DNSH Compliance",
            "Governance Assessment",
            "EU Taxonomy Alignment",
            "PAI Summary",
            "Impact Highlights",
            "Downgrade Risk Assessment",
            "Carbon Trajectory",
        ],
    },
    "pai_mandatory_report": {
        "class": PAIMandatoryReportTemplate,
        "display_name": "PAI Mandatory Indicators Report",
        "description": (
            "All 18 mandatory PAI indicators report for Article 9 products. "
            "Covers climate/environment (PAI 1-6), social/governance "
            "(PAI 10-14), sovereigns (PAI 15-16), real estate (PAI 17-18), "
            "additional indicators, integration in decisions, engagement "
            "policies, and data quality assessment."
        ),
        "template_class": PAIMandatoryReportTemplate,
        "category": "regulatory",
        "scope": "entity-level",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Summary",
            "Mandatory PAI Indicators (1-18)",
            "Additional PAI Indicators",
            "Integration in Investment Decisions",
            "Engagement Policies",
            "Data Quality",
        ],
    },
    "carbon_trajectory_report": {
        "class": CarbonTrajectoryReportTemplate,
        "display_name": "Carbon Trajectory Report",
        "description": (
            "Carbon intensity trajectory to 2050 report for Article 9 "
            "products. Covers intensity trajectory, Paris alignment "
            "assessment, SBT coverage, Implied Temperature Rise (ITR), "
            "carbon budget consumption, and net zero progress tracking."
        ),
        "template_class": CarbonTrajectoryReportTemplate,
        "category": "analytics",
        "scope": "trajectory",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Carbon Intensity Trajectory",
            "Paris Agreement Alignment",
            "Science Based Targets Coverage",
            "Implied Temperature Rise",
            "Carbon Budget",
            "Net Zero Progress",
        ],
    },
    "audit_trail_report": {
        "class": AuditTrailReportTemplate,
        "display_name": "Audit Trail Report",
        "description": (
            "Comprehensive audit trail and data provenance report for "
            "Article 9 products. Covers provenance hashes with SHA-256, "
            "data source inventory, methodology references, data quality "
            "flags, version history, and sign-off tracking."
        ),
        "template_class": AuditTrailReportTemplate,
        "category": "audit",
        "scope": "internal",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Provenance Hashes",
            "Data Source Inventory",
            "Methodology References",
            "Data Quality Flags",
            "Version History",
            "Sign-Off Tracking",
        ],
    },
}


class TemplateRegistry:
    """
    Registry for discovering, instantiating, and rendering SFDR Article 9 templates.

    Provides a centralized catalog of all 8 SFDR Article 9 Pack templates
    with metadata for programmatic discovery, filtering, instantiation,
    and a unified render() method supporting markdown, html, and json.

    Attributes:
        config: Optional global configuration passed to all templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("impact_report")
        >>> markdown = template.render_markdown(data)

        >>> html = registry.render("annex_iii_precontractual", data, "html")

        >>> regulatory = registry.list_templates(category="regulatory")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize TemplateRegistry.

        Args:
            config: Optional global configuration dictionary passed to
                template constructors when instantiated via get_template()
                or render().
        """
        self.config = config or {}
        self._catalog: Dict[str, Dict[str, Any]] = TEMPLATE_CATALOG.copy()

    def render(
        self,
        template_name: str,
        data: Dict[str, Any],
        format: str = "markdown",
        config: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Render a template by name in the specified format.

        Args:
            template_name: Template key (e.g. "annex_iii_precontractual").
            data: Report data dictionary.
            format: Output format - "markdown", "html", or "json".
            config: Optional per-render configuration override.

        Returns:
            Rendered content as string (markdown/html) or dict (json).

        Raises:
            KeyError: If template_name is not found.
            ValueError: If format is not supported.
        """
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        template = self.get_template(template_name, config=config)

        if format == "markdown":
            return template.render_markdown(data)
        elif format == "html":
            return template.render_html(data)
        else:
            return template.render_json(data)

    def list_templates(
        self,
        category: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available templates with optional filtering.

        Args:
            category: Filter by category (regulatory, analytics,
                audit).
            scope: Filter by scope (pre-contractual, periodic, impact,
                benchmark, dashboard, entity-level, trajectory, internal).

        Returns:
            List of template metadata dictionaries.
        """
        result: List[Dict[str, Any]] = []

        for key, meta in self._catalog.items():
            if category and meta.get("category") != category:
                continue
            if scope and meta.get("scope") != scope:
                continue

            result.append({
                "key": key,
                "display_name": meta["display_name"],
                "description": meta["description"],
                "category": meta["category"],
                "scope": meta["scope"],
                "version": meta["version"],
                "supported_formats": meta.get("supported_formats", list(SUPPORTED_FORMATS)),
                "sections": meta.get("sections", []),
            })

        return result

    def get_template(
        self,
        template_key: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Instantiate and return a template by its registry key.

        Args:
            template_key: Template identifier (e.g. "annex_iii_precontractual").
            config: Optional per-template configuration. If None, uses
                the registry-level config.

        Returns:
            Instantiated template object.

        Raises:
            KeyError: If template_key is not found.
        """
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Template '{template_key}' not found. Available: {available}"
            )

        template_cls = self._catalog[template_key]["class"]
        effective_config = config if config is not None else self.config
        return template_cls(config=effective_config)

    def get_template_metadata(self, template_key: str) -> Dict[str, Any]:
        """
        Get metadata for a specific template without instantiating it.

        Args:
            template_key: Template identifier.

        Returns:
            Dictionary with template metadata.

        Raises:
            KeyError: If template_key is not found.
        """
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Template '{template_key}' not found. Available: {available}"
            )

        meta = self._catalog[template_key]
        return {
            "key": template_key,
            "display_name": meta["display_name"],
            "description": meta["description"],
            "category": meta["category"],
            "scope": meta["scope"],
            "version": meta["version"],
            "supported_formats": meta.get("supported_formats", list(SUPPORTED_FORMATS)),
            "sections": meta.get("sections", []),
        }

    def get_all_template_keys(self) -> List[str]:
        """
        Get all registered template keys.

        Returns:
            Sorted list of template key strings.
        """
        return sorted(self._catalog.keys())

    def has_template(self, template_key: str) -> bool:
        """
        Check if a template key is registered.

        Args:
            template_key: Template identifier to check.

        Returns:
            True if the template exists in the registry.
        """
        return template_key in self._catalog

    @property
    def template_count(self) -> int:
        """Return the number of registered templates."""
        return len(self._catalog)

    @property
    def pack_id(self) -> str:
        """Return the pack identifier."""
        return "PACK-011"

    @property
    def pack_name(self) -> str:
        """Return the human-readable pack name."""
        return "SFDR Article 9"


# Module-level exports
__all__ = [
    # Template classes
    "AnnexIIIPrecontractualTemplate",
    "AnnexVPeriodicTemplate",
    "ImpactReportTemplate",
    "BenchmarkMethodologyTemplate",
    "SustainableDashboardTemplate",
    "PAIMandatoryReportTemplate",
    "CarbonTrajectoryReportTemplate",
    "AuditTrailReportTemplate",
    # Data models - Annex III
    "AnnexIIIPrecontractualData",
    "Article9ProductInfo",
    "SustainableObjective",
    "Article9Strategy",
    "Article9Proportions",
    "Article9TaxonomyAlignment",
    "Article9DNSHApproach",
    "Article9DataSources",
    "Article9Benchmark",
    # Data models - Impact Report
    "ImpactReportData",
    "ImpactFundInfo",
    "TheoryOfChange",
    "EnvironmentalKPI",
    "SocialKPI",
    "SDGContribution",
    "YoYComparison",
    "AdditionalityAssessment",
    # Data models - Benchmark Methodology
    "BenchmarkMethodologyData",
    "BenchmarkFundInfo",
    "BenchmarkSelection",
    "CarbonIntensityComparison",
    "DecarbonizationTrajectory",
    "ExclusionCompliance",
    "TrackingErrorAnalysis",
    # Data models - Sustainable Dashboard
    "SustainableDashboardData",
    "DashboardFundInfo",
    "SustainableMetrics",
    "DNSHMetrics",
    "GovernanceMetrics",
    "TaxonomyMetrics",
    "PAISummaryMetrics",
    "ImpactHighlights",
    "DowngradeRisk",
    "CarbonTrajectoryDashboard",
    # Data models - PAI Mandatory Report
    "PAIMandatoryReportData",
    "PAIEntityInfo",
    "MandatoryPAIIndicator",
    "AdditionalPAIIndicator",
    "PAIIntegrationDescription",
    "PAIEngagement",
    "PAIDataQuality",
    # Data models - Carbon Trajectory Report
    "CarbonTrajectoryReportData",
    "TrajectoryFundInfo",
    "IntensityTrajectory",
    "ParisAlignmentAssessment",
    "SBTCoverage",
    "ImpliedTemperatureRise",
    "CarbonBudget",
    "NetZeroProgress",
    # Data models - Audit Trail Report
    "AuditTrailReportData",
    "AuditReportInfo",
    "ProvenanceRecord",
    "DataSourceEntry",
    "MethodologyReference",
    "DataQualityFlag",
    "VersionHistoryEntry",
    "SignOffRecord",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    "SUPPORTED_FORMATS",
]
