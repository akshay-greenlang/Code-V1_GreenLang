"""
SFDR Article 8 Pack Templates - PACK-010

This module exports all 8 SFDR Article 8 report template classes and provides
a TemplateRegistry for programmatic template discovery, rendering, and
instantiation.

Templates:
    - AnnexIIPrecontractualTemplate: SFDR RTS Annex II pre-contractual disclosure
    - AnnexIVPeriodicTemplate: SFDR RTS Annex IV periodic reporting
    - AnnexIIIWebsiteTemplate: SFDR RTS Annex III website disclosure
    - PAIStatementTemplate: Principal Adverse Impact indicator statement
    - PortfolioESGDashboardTemplate: Interactive-style ESG metrics dashboard
    - TaxonomyAlignmentReportTemplate: EU Taxonomy alignment analysis
    - ExecutiveSummaryTemplate: Board-level executive summary
    - AuditTrailReportTemplate: Data lineage and provenance audit trail

Supported Formats:
    - Markdown (GitHub-flavored tables)
    - HTML (self-contained with inline CSS)
    - JSON (structured, machine-readable)

Example:
    >>> from packs.eu_compliance.PACK_010_sfdr_article_8.templates import (
    ...     AnnexIIPrecontractualTemplate,
    ...     TemplateRegistry,
    ... )
    >>> template = AnnexIIPrecontractualTemplate()
    >>> markdown = template.render_markdown(data)

    >>> registry = TemplateRegistry()
    >>> names = registry.list_templates()
    >>> template = registry.get_template("annex_ii_precontractual")
    >>> html = registry.render("annex_ii_precontractual", data, "html")
"""

from typing import Any, Dict, List, Optional, Type, Union

from .annex_ii_precontractual import AnnexIIPrecontractualTemplate
from .annex_iv_periodic import AnnexIVPeriodicTemplate
from .annex_iii_website import AnnexIIIWebsiteTemplate
from .pai_statement_template import PAIStatementTemplate
from .portfolio_esg_dashboard import PortfolioESGDashboardTemplate
from .taxonomy_alignment_report import TaxonomyAlignmentReportTemplate
from .executive_summary import ExecutiveSummaryTemplate
from .audit_trail_report import AuditTrailReportTemplate

# Re-export key data models for convenience
from .annex_ii_precontractual import (
    PrecontractualData,
    ProductInfo,
    ESCharacteristics,
    AssetAllocation,
    InvestmentStrategy,
    TaxonomyDisclosure,
    DNSHApproach,
    DataSources,
)
from .annex_iv_periodic import (
    PeriodicData,
    ReportingPeriod,
    CharacteristicAttainment,
    TopInvestment,
    ProportionBreakdown,
    PAISummary,
)
from .annex_iii_website import WebsiteDisclosureData
from .pai_statement_template import PAIStatementData, PAIIndicatorData
from .portfolio_esg_dashboard import DashboardData
from .taxonomy_alignment_report import AlignmentReportData
from .executive_summary import ExecutiveSummaryData
from .audit_trail_report import AuditTrailData


# Type alias for any template class in this pack
TemplateClass = Union[
    Type[AnnexIIPrecontractualTemplate],
    Type[AnnexIVPeriodicTemplate],
    Type[AnnexIIIWebsiteTemplate],
    Type[PAIStatementTemplate],
    Type[PortfolioESGDashboardTemplate],
    Type[TaxonomyAlignmentReportTemplate],
    Type[ExecutiveSummaryTemplate],
    Type[AuditTrailReportTemplate],
]

# Supported output formats
SUPPORTED_FORMATS = ("markdown", "html", "json")

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "annex_ii_precontractual": {
        "class": AnnexIIPrecontractualTemplate,
        "display_name": "Pre-contractual Disclosure (Annex II)",
        "description": (
            "SFDR RTS Annex II pre-contractual disclosure template for "
            "Article 8 products. Covers 12 sections including product "
            "identification, E/S characteristics, asset allocation, "
            "taxonomy alignment, DNSH approach, data sources, and "
            "engagement policies."
        ),
        "template_class": AnnexIIPrecontractualTemplate,
        "category": "regulatory",
        "scope": "pre-contractual",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Product Identification",
            "E/S Characteristics",
            "Sustainable Investment Objective Statement",
            "Investment Strategy",
            "Proportion of Investments",
            "Minimum Sustainable Investment",
            "EU Taxonomy Alignment",
            "DNSH Approach",
            "Data Sources and Processing",
            "Limitations",
            "Due Diligence",
            "Engagement Policies",
        ],
    },
    "annex_iv_periodic": {
        "class": AnnexIVPeriodicTemplate,
        "display_name": "Periodic Report (Annex IV)",
        "description": (
            "SFDR RTS Annex IV periodic reporting template. Reports on "
            "how E/S characteristics were attained during the reference "
            "period, including top investments, proportion breakdowns, "
            "PAI indicators, actions taken, and benchmark comparison."
        ),
        "template_class": AnnexIVPeriodicTemplate,
        "category": "regulatory",
        "scope": "periodic",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Product Identification",
            "E/S Characteristics Attainment",
            "Top Investments",
            "Proportion of Sustainability-Related Investments",
            "PAI Indicators",
            "Actions Taken",
            "Comparison to Previous Period",
            "Benchmark Comparison",
        ],
    },
    "annex_iii_website": {
        "class": AnnexIIIWebsiteTemplate,
        "display_name": "Website Disclosure (Annex III)",
        "description": (
            "SFDR RTS Annex III website disclosure template. Provides "
            "the 12 mandatory website disclosure sections including "
            "summary, investment strategy, proportions, monitoring, "
            "methodologies, data sources, limitations, due diligence, "
            "and engagement policies."
        ),
        "template_class": AnnexIIIWebsiteTemplate,
        "category": "regulatory",
        "scope": "website",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Summary",
            "Sustainable Investment Objective",
            "E/S Characteristics",
            "Investment Strategy",
            "Proportion of Investments",
            "Monitoring of E/S Characteristics",
            "Methodologies",
            "Data Sources and Processing",
            "Limitations",
            "Due Diligence",
            "Engagement Policies",
            "Designated Reference Benchmark",
        ],
    },
    "pai_statement": {
        "class": PAIStatementTemplate,
        "display_name": "PAI Statement",
        "description": (
            "Principal Adverse Impact indicator statement template. "
            "Covers all 18 mandatory PAI indicators from SFDR RTS "
            "Annex I Table 1 with year-over-year comparisons, "
            "explanations, actions taken, and engagement outcomes."
        ),
        "template_class": PAIStatementTemplate,
        "category": "regulatory",
        "scope": "entity-level",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Entity Information",
            "PAI Indicator Summary",
            "Climate and Environment (PAI 1-9)",
            "Social and Governance (PAI 10-14)",
            "Sovereigns (PAI 15-16)",
            "Real Estate (PAI 17-18)",
            "Additional Indicators",
            "Narrative Disclosures",
        ],
    },
    "portfolio_esg_dashboard": {
        "class": PortfolioESGDashboardTemplate,
        "display_name": "Portfolio ESG Dashboard",
        "description": (
            "Interactive-style ESG metrics dashboard with fund overview, "
            "ESG score gauges, taxonomy alignment, carbon metrics, "
            "sector allocation, PAI summary tiles, commitment tracker, "
            "and compliance alerts."
        ),
        "template_class": PortfolioESGDashboardTemplate,
        "category": "analytics",
        "scope": "portfolio",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Fund Overview",
            "ESG Scores",
            "Taxonomy Alignment",
            "Carbon Metrics",
            "Sector Allocation",
            "PAI Summary",
            "Commitment Tracker",
            "Compliance Alerts",
        ],
    },
    "taxonomy_alignment_report": {
        "class": TaxonomyAlignmentReportTemplate,
        "display_name": "Taxonomy Alignment Report",
        "description": (
            "Detailed EU Taxonomy alignment analysis with alignment "
            "summary, objective breakdowns, fossil gas/nuclear disclosure, "
            "eligible vs. aligned comparison, commitment adherence, "
            "and top aligned holdings."
        ),
        "template_class": TaxonomyAlignmentReportTemplate,
        "category": "analytics",
        "scope": "taxonomy",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Alignment Summary",
            "Breakdown by Environmental Objective",
            "Fossil Gas and Nuclear Disclosure",
            "Eligible vs. Aligned",
            "Commitment Adherence",
            "Top Aligned Holdings",
            "Methodology Notes",
        ],
    },
    "executive_summary": {
        "class": ExecutiveSummaryTemplate,
        "display_name": "Executive Summary",
        "description": (
            "Board-level executive summary with fund classification card, "
            "key metrics, compliance traffic lights, risk flags, "
            "strategic recommendations, and regulatory outlook."
        ),
        "template_class": ExecutiveSummaryTemplate,
        "category": "governance",
        "scope": "board",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Fund Classification",
            "Key Metrics",
            "Compliance Status",
            "Risk Flags",
            "Strategic Recommendations",
            "Regulatory Outlook",
        ],
    },
    "audit_trail_report": {
        "class": AuditTrailReportTemplate,
        "display_name": "Audit Trail Report",
        "description": (
            "Comprehensive audit trail documentation with data quality "
            "overview, data lineage diagrams, calculation provenance "
            "with SHA-256 hashes, methodology references, data source "
            "inventory, estimation log, and assumptions registry."
        ),
        "template_class": AuditTrailReportTemplate,
        "category": "audit",
        "scope": "internal",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Data Quality Overview",
            "Data Lineage",
            "Calculation Provenance",
            "Methodology References",
            "Data Source Inventory",
            "Estimation Log",
            "Assumptions Registry",
            "Auditor Notes",
        ],
    },
}


class TemplateRegistry:
    """
    Registry for discovering, instantiating, and rendering SFDR Article 8 templates.

    Provides a centralized catalog of all 8 SFDR Article 8 Pack templates
    with metadata for programmatic discovery, filtering, instantiation,
    and a unified render() method supporting markdown, html, and json.

    Attributes:
        config: Optional global configuration passed to all templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("executive_summary")
        >>> markdown = template.render_markdown(data)

        >>> html = registry.render("annex_ii_precontractual", data, "html")

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
            template_name: Template key (e.g. "annex_ii_precontractual").
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
                governance, audit).
            scope: Filter by scope (pre-contractual, periodic, website,
                entity-level, portfolio, taxonomy, board, internal).

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
            template_key: Template identifier (e.g. "annex_ii_precontractual").
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
        return "PACK-010"

    @property
    def pack_name(self) -> str:
        """Return the human-readable pack name."""
        return "SFDR Article 8"


# Module-level exports
__all__ = [
    # Template classes
    "AnnexIIPrecontractualTemplate",
    "AnnexIVPeriodicTemplate",
    "AnnexIIIWebsiteTemplate",
    "PAIStatementTemplate",
    "PortfolioESGDashboardTemplate",
    "TaxonomyAlignmentReportTemplate",
    "ExecutiveSummaryTemplate",
    "AuditTrailReportTemplate",
    # Data models
    "PrecontractualData",
    "ProductInfo",
    "ESCharacteristics",
    "AssetAllocation",
    "InvestmentStrategy",
    "TaxonomyDisclosure",
    "DNSHApproach",
    "DataSources",
    "PeriodicData",
    "ReportingPeriod",
    "CharacteristicAttainment",
    "TopInvestment",
    "ProportionBreakdown",
    "PAISummary",
    "WebsiteDisclosureData",
    "PAIStatementData",
    "PAIIndicatorData",
    "DashboardData",
    "AlignmentReportData",
    "ExecutiveSummaryData",
    "AuditTrailData",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    "SUPPORTED_FORMATS",
]
