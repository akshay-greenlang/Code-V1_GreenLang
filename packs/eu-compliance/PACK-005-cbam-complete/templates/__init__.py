"""
CBAM Complete Pack Templates - PACK-005

This module exports all 6 CBAM Complete report template classes and provides a
TemplateRegistry for programmatic template discovery, rendering, and instantiation.

Templates:
    - CertificatePortfolioReport: Certificate inventory, cost analysis, expiry tracking
    - GroupConsolidationReport: Multi-entity group consolidation and cost allocation
    - SourcingScenarioAnalysis: Scenario modeling, Monte Carlo, sensitivity analysis
    - CrossRegulationMappingReport: CBAM data reuse across 6 regulations
    - CustomsIntegrationReport: Customs-CBAM reconciliation, CN codes, EORI validation
    - AuditReadinessScorecard: Audit readiness scoring, findings, corrective actions

Supported Formats:
    - Markdown (GitHub-flavored tables)
    - HTML (self-contained with inline CSS)
    - JSON (structured, machine-readable)

Example:
    >>> from packs.eu_compliance.PACK_005_cbam_complete.templates import (
    ...     CertificatePortfolioReport,
    ...     TemplateRegistry,
    ... )
    >>> template = CertificatePortfolioReport()
    >>> markdown = template.render_markdown(data)

    >>> registry = TemplateRegistry()
    >>> names = registry.list_templates()
    >>> template = registry.get_template("certificate_portfolio_report")
    >>> html = registry.render("certificate_portfolio_report", data, "html")
"""

from typing import Any, Dict, List, Optional, Type, Union

from .certificate_portfolio_report import CertificatePortfolioReport
from .group_consolidation_report import GroupConsolidationReport
from .sourcing_scenario_analysis import SourcingScenarioAnalysis
from .cross_regulation_mapping_report import CrossRegulationMappingReport
from .customs_integration_report import CustomsIntegrationReport
from .audit_readiness_scorecard import AuditReadinessScorecard


# Type alias for any template class in this pack
TemplateClass = Union[
    Type[CertificatePortfolioReport],
    Type[GroupConsolidationReport],
    Type[SourcingScenarioAnalysis],
    Type[CrossRegulationMappingReport],
    Type[CustomsIntegrationReport],
    Type[AuditReadinessScorecard],
]

# Supported output formats
SUPPORTED_FORMATS = ("markdown", "html", "json")

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "certificate_portfolio_report": {
        "class": CertificatePortfolioReport,
        "name": "Certificate Portfolio Report",
        "description": (
            "Comprehensive certificate portfolio management including inventory "
            "tracking, cost analysis, expiry timelines, budget vs actual, quarterly "
            "holding compliance, purchase/surrender history, and ETS price trends."
        ),
        "category": "financial",
        "phase": "definitive",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Portfolio Summary",
            "Cost Analysis",
            "Expiry Timeline",
            "Budget vs Actual",
            "Quarterly Holding Compliance",
            "Purchase History",
            "Surrender History",
            "Price Trend",
        ],
    },
    "group_consolidation_report": {
        "class": GroupConsolidationReport,
        "name": "Group Consolidation Report",
        "description": (
            "Multi-entity group consolidation with entity hierarchies, per-entity "
            "breakdowns, consolidated obligations, cost allocation across subsidiaries, "
            "member state summaries, de minimis analysis, and financial guarantees."
        ),
        "category": "reporting",
        "phase": "definitive",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Group Overview",
            "Entity Hierarchy",
            "Per-Entity Breakdown",
            "Consolidated Obligation",
            "Cost Allocation",
            "Member State Summary",
            "De Minimis Analysis",
            "Financial Guarantee",
        ],
    },
    "sourcing_scenario_analysis": {
        "class": SourcingScenarioAnalysis,
        "name": "Sourcing Scenario Analysis",
        "description": (
            "Sourcing optimization with scenario comparisons, Monte Carlo simulations, "
            "sensitivity analysis (tornado diagram), supplier switching impacts, "
            "decarbonization ROI, ranked recommendations, and modeling assumptions."
        ),
        "category": "analytics",
        "phase": "both",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Current Profile",
            "Scenario Comparison",
            "Monte Carlo Results",
            "Sensitivity Analysis",
            "Supplier Switching Impact",
            "Decarbonization ROI",
            "Recommendations",
            "Assumptions",
        ],
    },
    "cross_regulation_mapping_report": {
        "class": CrossRegulationMappingReport,
        "name": "Cross-Regulation Mapping Report",
        "description": (
            "Maps CBAM data flows to 6 related regulations (CSRD, CDP, SBTi, "
            "EU Taxonomy, EU ETS, EUDR) with field-level mappings, data reuse "
            "statistics, efficiency metrics, and consistency checks."
        ),
        "category": "compliance",
        "phase": "both",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Data Flow Overview",
            "CSRD Mapping",
            "CDP Mapping",
            "SBTi Mapping",
            "EU Taxonomy Mapping",
            "EU ETS Mapping",
            "EUDR Mapping",
            "Data Reuse Statistics",
            "Consistency Checks",
        ],
    },
    "customs_integration_report": {
        "class": CustomsIntegrationReport,
        "name": "Customs Integration Report",
        "description": (
            "Customs-CBAM integration covering import summaries by CN code, "
            "CBAM applicability assessment, TARIC validation, anti-circumvention "
            "flags, customs procedure breakdown, duty+CBAM cost, SAD reconciliation, "
            "and EORI validation."
        ),
        "category": "customs",
        "phase": "both",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Import Summary",
            "CBAM Applicability",
            "CN Code Validation",
            "Anti-Circumvention Flags",
            "Customs Procedure Breakdown",
            "Duty + CBAM Cost",
            "SAD Reconciliation",
            "EORI Validation",
        ],
    },
    "audit_readiness_scorecard": {
        "class": AuditReadinessScorecard,
        "name": "Audit Readiness Scorecard",
        "description": (
            "Comprehensive audit readiness assessment with overall and category "
            "scores, evidence completeness, unresolved findings, corrective action "
            "tracking, verifier engagement, NCA correspondence, penalty exposure "
            "estimates, and prioritized action items."
        ),
        "category": "audit",
        "phase": "definitive",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Overall Readiness Score",
            "Category Scores",
            "Evidence Completeness",
            "Unresolved Findings",
            "Corrective Action Status",
            "Verifier Engagement",
            "NCA Correspondence",
            "Penalty Exposure",
            "Action Items",
        ],
    },
}


class TemplateRegistry:
    """
    Registry for discovering, instantiating, and rendering CBAM Complete templates.

    Provides a centralized catalog of all available CBAM Complete Pack templates
    with metadata for programmatic discovery, filtering, instantiation, and a
    unified render() method that supports markdown, html, and json output.

    Attributes:
        config: Optional global configuration passed to all templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("audit_readiness_scorecard")
        >>> markdown = template.render_markdown(data)

        >>> # Use unified render method
        >>> html = registry.render("certificate_portfolio_report", data, "html")

        >>> # Filter by category
        >>> financial = registry.list_templates(category="financial")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize TemplateRegistry.

        Args:
            config: Optional global configuration dictionary that will be
                passed to template constructors when instantiated via
                get_template() or render().
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

        This is the primary entry point for rendering reports. It instantiates
        the template and calls the appropriate render method based on the
        requested format.

        Args:
            template_name: Template key (e.g. "certificate_portfolio_report").
            data: Report data dictionary.
            format: Output format - "markdown", "html", or "json".
            config: Optional per-render configuration override.

        Returns:
            Rendered content as string (markdown/html) or dict (json).

        Raises:
            KeyError: If template_name is not found in the registry.
            ValueError: If format is not supported.
        """
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
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
        phase: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available templates with optional filtering.

        Args:
            category: Filter by category (financial, reporting, analytics,
                compliance, customs, audit).
            phase: Filter by phase (transitional, definitive, both).

        Returns:
            List of template metadata dictionaries containing:
                - key (str): Template identifier for get_template()/render()
                - name (str): Human-readable template name
                - description (str): Template description
                - category (str): Template category
                - phase (str): CBAM phase applicability
                - version (str): Template version
                - supported_formats (list[str]): Supported output formats
                - sections (list[str]): Report section names
        """
        result: List[Dict[str, Any]] = []

        for key, meta in self._catalog.items():
            if category and meta.get("category") != category:
                continue

            if phase:
                template_phase = meta.get("phase", "both")
                if template_phase != "both" and template_phase != phase:
                    continue

            result.append({
                "key": key,
                "name": meta["name"],
                "description": meta["description"],
                "category": meta["category"],
                "phase": meta["phase"],
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
            template_key: Template identifier (e.g. "certificate_portfolio_report").
            config: Optional per-template configuration. If None, uses
                the registry-level config.

        Returns:
            Instantiated template object with render_markdown(), render_html(),
            and render_json() methods.

        Raises:
            KeyError: If template_key is not found in the registry.
        """
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Template '{template_key}' not found. "
                f"Available templates: {available}"
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
            Dictionary with template metadata (name, description, category,
            phase, version, supported_formats, sections).

        Raises:
            KeyError: If template_key is not found in the registry.
        """
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Template '{template_key}' not found. "
                f"Available templates: {available}"
            )

        meta = self._catalog[template_key]
        return {
            "key": template_key,
            "name": meta["name"],
            "description": meta["description"],
            "category": meta["category"],
            "phase": meta["phase"],
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
        return "PACK-005"

    @property
    def pack_name(self) -> str:
        """Return the human-readable pack name."""
        return "CBAM Complete"


# Module-level exports
__all__ = [
    # Template classes
    "CertificatePortfolioReport",
    "GroupConsolidationReport",
    "SourcingScenarioAnalysis",
    "CrossRegulationMappingReport",
    "CustomsIntegrationReport",
    "AuditReadinessScorecard",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    "SUPPORTED_FORMATS",
]
