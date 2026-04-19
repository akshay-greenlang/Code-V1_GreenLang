# -*- coding: utf-8 -*-
"""
PACK-030 Net Zero Reporting Pack - Report Templates
====================================================

This package provides 15 report templates for the PACK-030 Net Zero
Reporting Pack, designed for organisations disclosing net-zero progress
across multiple regulatory frameworks (SBTi, CDP, TCFD, GRI, ISSB, SEC,
CSRD). Templates cover framework-specific disclosures, stakeholder
dashboards, product carbon footprints, and assurance evidence packages.
All templates support multi-format output (Markdown, HTML, JSON, PDF)
with SHA-256 provenance hashing.

Templates:
    1.  SBTiProgressTemplate               - SBTi target progress with scope-level tracking
    2.  CDPGovernanceTemplate              - CDP C0-C2 governance & risk/opportunity disclosure
    3.  CDPEmissionsTemplate               - CDP C4-C7 targets, methodology, emissions
    4.  TCFDGovernanceTemplate             - TCFD governance pillar (board + management)
    5.  TCFDStrategyTemplate               - TCFD strategy pillar with scenario analysis
    6.  TCFDRiskTemplate                   - TCFD risk management pillar
    7.  TCFDMetricsTemplate                - TCFD metrics & targets pillar
    8.  GRI305Template                     - GRI 305-1 through 305-7 emissions disclosures
    9.  ISSBS2Template                     - IFRS S2 climate-related disclosures
    10. SECClimateTemplate                 - SEC Reg S-K climate disclosure (Items 1502-1506)
    11. CSRDE1Template                     - CSRD ESRS E1 climate change (E1-1 to E1-9)
    12. InvestorDashboardTemplate          - Investor-focused climate dashboard
    13. RegulatorDashboardTemplate         - Regulator compliance dashboard (CSRD + SEC)
    14. CustomerCarbonTemplate             - Customer product carbon footprint report
    15. AssuranceEvidenceTemplate          - ISO 14064-3 assurance evidence package

Usage:
    >>> from packs.net_zero.PACK_030_net_zero_reporting.templates import (
    ...     TemplateRegistry,
    ...     SBTiProgressTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("sbti_progress")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("cdp_emissions", data, format="html")

    >>> # Filter by category
    >>> tcfd_templates = registry.get_by_category("tcfd")

    >>> # Multi-format batch render
    >>> for fmt in ["markdown", "html", "json"]:
    ...     output = registry.render("investor_dashboard", data, format=fmt)

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

__version__ = "30.0.0"
__pack_id__ = "PACK-030"
__pack_name__ = "Net Zero Reporting Pack"

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .sbti_progress_template import SBTiProgressTemplate
except ImportError:
    SBTiProgressTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SBTiProgressTemplate")

try:
    from .cdp_governance_template import CDPGovernanceTemplate
except ImportError:
    CDPGovernanceTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CDPGovernanceTemplate")

try:
    from .cdp_emissions_template import CDPEmissionsTemplate
except ImportError:
    CDPEmissionsTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CDPEmissionsTemplate")

try:
    from .tcfd_governance_template import TCFDGovernanceTemplate
except ImportError:
    TCFDGovernanceTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TCFDGovernanceTemplate")

try:
    from .tcfd_strategy_template import TCFDStrategyTemplate
except ImportError:
    TCFDStrategyTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TCFDStrategyTemplate")

try:
    from .tcfd_risk_template import TCFDRiskTemplate
except ImportError:
    TCFDRiskTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TCFDRiskTemplate")

try:
    from .tcfd_metrics_template import TCFDMetricsTemplate
except ImportError:
    TCFDMetricsTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TCFDMetricsTemplate")

try:
    from .gri_305_template import GRI305Template
except ImportError:
    GRI305Template = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GRI305Template")

try:
    from .issb_s2_template import ISSBS2Template
except ImportError:
    ISSBS2Template = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ISSBS2Template")

try:
    from .sec_climate_template import SECClimateTemplate
except ImportError:
    SECClimateTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SECClimateTemplate")

try:
    from .csrd_e1_template import CSRDE1Template
except ImportError:
    CSRDE1Template = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CSRDE1Template")

try:
    from .investor_dashboard_template import InvestorDashboardTemplate
except ImportError:
    InvestorDashboardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import InvestorDashboardTemplate")

try:
    from .regulator_dashboard_template import RegulatorDashboardTemplate
except ImportError:
    RegulatorDashboardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import RegulatorDashboardTemplate")

try:
    from .customer_carbon_template import CustomerCarbonTemplate
except ImportError:
    CustomerCarbonTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CustomerCarbonTemplate")

try:
    from .assurance_evidence_template import AssuranceEvidenceTemplate
except ImportError:
    AssuranceEvidenceTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AssuranceEvidenceTemplate")


__all__ = [
    # Version info
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Template classes
    "SBTiProgressTemplate",
    "CDPGovernanceTemplate",
    "CDPEmissionsTemplate",
    "TCFDGovernanceTemplate",
    "TCFDStrategyTemplate",
    "TCFDRiskTemplate",
    "TCFDMetricsTemplate",
    "GRI305Template",
    "ISSBS2Template",
    "SECClimateTemplate",
    "CSRDE1Template",
    "InvestorDashboardTemplate",
    "RegulatorDashboardTemplate",
    "CustomerCarbonTemplate",
    "AssuranceEvidenceTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "sbti_progress",
        "class": SBTiProgressTemplate,
        "description": (
            "SBTi target progress report with scope-level emissions tracking, "
            "base year comparison, variance analysis with RAG scoring, milestone "
            "tracking, initiative deployment status, forward-looking projections, "
            "XBRL tagging, and audit trail. Covers near-term and long-term SBTi "
            "1.5C-aligned targets."
        ),
        "category": "sbti",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "cdp_governance",
        "class": CDPGovernanceTemplate,
        "description": (
            "CDP Climate Change governance disclosure covering C0 (Introduction), "
            "C1 (Governance), and C2 (Risks and Opportunities) modules. Includes "
            "board oversight, management role, climate incentives, governance "
            "questions, risk categories, opportunity categories, and CDP scoring "
            "readiness assessment."
        ),
        "category": "cdp",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "cdp_emissions",
        "class": CDPEmissionsTemplate,
        "description": (
            "CDP Climate Change emissions disclosure covering C4 (Targets), "
            "C5 (Emissions Methodology), C6 (Emissions - Scope 1/2), and "
            "C7 (Emissions - Scope 3) modules. Includes all 15 Scope 3 "
            "categories, GHG gas breakdown, methodology notes, completeness "
            "scoring, and CDP A-list optimization guidance."
        ),
        "category": "cdp",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "tcfd_governance",
        "class": TCFDGovernanceTemplate,
        "description": (
            "TCFD Governance pillar disclosure covering board oversight of "
            "climate-related risks and opportunities, management's role in "
            "assessing and managing climate risks, climate competence matrix, "
            "governance effectiveness scoring across 8 criteria, and "
            "governance improvement recommendations."
        ),
        "category": "tcfd",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "tcfd_strategy",
        "class": TCFDStrategyTemplate,
        "description": (
            "TCFD Strategy pillar disclosure with climate-related risks and "
            "opportunities identification, scenario analysis (IEA NZE 1.5C, "
            "APS 2C, STEPS 4C), strategic resilience assessment, transition "
            "plan summary, and financial impact quantification including "
            "carbon prices and revenue-at-risk analysis."
        ),
        "category": "tcfd",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "tcfd_risk",
        "class": TCFDRiskTemplate,
        "description": (
            "TCFD Risk Management pillar disclosure covering risk identification "
            "process, assessment framework with likelihood and impact scales, "
            "climate risk register, mitigation strategies, ERM integration, "
            "risk appetite statement, and emerging risk horizon scanning."
        ),
        "category": "tcfd",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "tcfd_metrics",
        "class": TCFDMetricsTemplate,
        "description": (
            "TCFD Metrics and Targets pillar disclosure with GHG emissions "
            "by scope, intensity metrics, emissions targets and progress, "
            "carbon pricing impact, executive remuneration linkage, year-over-year "
            "trend analysis, and cross-industry climate metrics."
        ),
        "category": "tcfd",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "gri_305",
        "class": GRI305Template,
        "description": (
            "GRI 305 Emissions disclosures covering 305-1 (Direct Scope 1), "
            "305-2 (Indirect Scope 2), 305-3 (Other Indirect Scope 3), "
            "305-4 (Intensity), 305-5 (Reduction), 305-6 (ODS), and "
            "305-7 (NOx/SOx). Includes GRI Content Index, methodology "
            "notes, and multi-year trend analysis."
        ),
        "category": "gri",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "issb_s2",
        "class": ISSBS2Template,
        "description": (
            "IFRS S2 Climate-Related Disclosures covering paragraphs 5-37 "
            "structured by governance, strategy, risk management, and "
            "metrics/targets. Includes cross-industry metrics, SASB "
            "industry-specific metrics, transition plan, resilience "
            "assessment, and ISSB paragraph mapping."
        ),
        "category": "issb",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "sec_climate",
        "class": SECClimateTemplate,
        "description": (
            "SEC climate disclosure covering Reg S-K Items 1502-1506 "
            "(governance, strategy, risk management, GHG emissions, "
            "targets). Includes 10-K integration mapping, attestation "
            "requirements, XBRL/iXBRL tagging with us-gaap taxonomy, "
            "and filing readiness assessment."
        ),
        "category": "sec",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "csrd_e1",
        "class": CSRDE1Template,
        "description": (
            "CSRD ESRS E1 Climate Change disclosure covering all 9 "
            "requirements (E1-1 through E1-9): transition plan, policies, "
            "actions, targets, energy consumption, emissions, removals, "
            "internal carbon pricing, and financial effects. Includes "
            "compliance scoring and ESRS digital taxonomy tags."
        ),
        "category": "csrd",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "investor_dashboard",
        "class": InvestorDashboardTemplate,
        "description": (
            "Investor-focused climate dashboard with key ESG/climate KPIs, "
            "TCFD alignment status across 4 pillars, emissions performance "
            "and trends, targets and progress tracking, scenario analysis "
            "summary, financial materiality assessment, ESG ratings, and "
            "climate risk heatmap."
        ),
        "category": "stakeholder",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "regulator_dashboard",
        "class": RegulatorDashboardTemplate,
        "description": (
            "Regulator compliance dashboard with CSRD ESRS E1 and SEC "
            "Reg S-K coverage tracking, mandatory disclosure checklist, "
            "data quality indicators, filing timeline and deadlines, "
            "audit trail and evidence links, enforcement risk assessment, "
            "and cross-framework disclosure mapping."
        ),
        "category": "stakeholder",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "customer_carbon",
        "class": CustomerCarbonTemplate,
        "description": (
            "Customer-facing product carbon footprint report covering "
            "lifecycle stage breakdown, supply chain emissions, use-phase "
            "emissions (Scope 3 Cat 11), end-of-life treatment (Scope 3 "
            "Cat 12), reduction initiatives, carbon labeling data (ISO "
            "14067, PAS 2050), and customer engagement metrics."
        ),
        "category": "stakeholder",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
    {
        "name": "assurance_evidence",
        "class": AssuranceEvidenceTemplate,
        "description": (
            "ISO 14064-3 aligned assurance evidence package with SHA-256 "
            "provenance hash chains, data lineage diagrams, methodology "
            "documentation, control matrix, calculation trails, 5-tier "
            "evidence hierarchy, materiality assessment, data quality "
            "scoring, 15-item readiness checklist, and ISO 14064-3 "
            "workpaper summaries."
        ),
        "category": "assurance",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "30.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-030 Net Zero Reporting Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 15 net-zero reporting templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON/PDF.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 15
        >>> template = registry.get("sbti_progress")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}

        for defn in TEMPLATE_CATALOG:
            if defn["class"] is not None:
                self._templates[defn["name"]] = defn

        logger.info(
            "PACK-030 TemplateRegistry initialized with %d templates",
            len(self._templates),
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates with metadata.

        Returns:
            List of template info dicts with name, description,
            category, formats, and version.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["class"] is not None
        ]

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List of template name strings.
        """
        return [defn["name"] for defn in TEMPLATE_CATALOG if defn["class"] is not None]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get a template instance by name.

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'sbti_progress').
            config: Optional configuration overrides.

        Returns:
            Template instance with render_markdown, render_html,
            render_json, and render_pdf methods.

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

    def get_template(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Alias for get(). Get a template instance by name.

        Args:
            name: Template name.
            config: Optional configuration overrides.

        Returns:
            Template instance.
        """
        return self.get(name, config)

    def render(
        self,
        template_name: str,
        data: Dict[str, Any],
        format: str = "markdown",
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Render a template in the specified format.

        Convenience method that gets the template and renders in one call.

        Args:
            template_name: Template name.
            data: Report data dict.
            format: Output format ('markdown', 'html', 'json', 'pdf').
            config: Optional template configuration.

        Returns:
            Rendered content (str for markdown/html, dict for json,
            bytes for pdf).

        Raises:
            KeyError: If template name not found.
            ValueError: If format is not supported.
        """
        template = self.get(template_name, config)
        if format == "markdown":
            return template.render_markdown(data)
        elif format == "html":
            return template.render_html(data)
        elif format == "json":
            return template.render_json(data)
        elif format == "pdf":
            if hasattr(template, "render_pdf"):
                return template.render_pdf(data)
            raise ValueError(
                f"Template '{template_name}' does not support PDF format."
            )
        else:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Use 'markdown', 'html', 'json', or 'pdf'."
            )

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get metadata about a specific template.

        Args:
            name: Template name.

        Returns:
            Template info dict with name, description, category,
            formats, version, and class_name.

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

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string. Valid categories:
                'sbti', 'cdp', 'tcfd', 'gri', 'issb', 'sec',
                'csrd', 'stakeholder', 'assurance'.

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["category"] == category and defn["class"] is not None
        ]

    def has_template(self, name: str) -> bool:
        """
        Check if a template exists by name.

        Args:
            name: Template name to check.

        Returns:
            True if template exists.
        """
        return name in self._templates

    @property
    def template_count(self) -> int:
        """
        Return the number of registered templates.

        Returns:
            Template count.
        """
        return len(self._templates)

    @property
    def categories(self) -> List[str]:
        """
        Return list of unique template categories.

        Returns:
            Sorted list of category strings.
        """
        cats = set()
        for defn in TEMPLATE_CATALOG:
            if defn["class"] is not None:
                cats.add(defn["category"])
        return sorted(cats)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search templates by name or description keyword.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching template info dicts.
        """
        query_lower = query.lower()
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["class"] is not None
            and (
                query_lower in defn["name"].lower()
                or query_lower in defn["description"].lower()
            )
        ]

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return (
            f"TemplateRegistry(pack='PACK-030', templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )

    def __len__(self) -> int:
        """Return number of registered templates."""
        return self.template_count

    def __contains__(self, name: str) -> bool:
        """Check if template name is registered."""
        return self.has_template(name)

    def __iter__(self):
        """Iterate over template names."""
        return iter(self.list_template_names())
