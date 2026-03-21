# -*- coding: utf-8 -*-
"""
PACK-029 Interim Targets Pack - Report Templates
====================================================

This package provides 10 report templates for the PACK-029 Interim Targets
Pack, designed for organisations tracking, reporting, and disclosing interim
decarbonization targets aligned with SBTi 1.5C pathways. Templates cover
target summaries, annual progress, variance analysis, corrective actions,
quarterly dashboards, CDP/TCFD disclosure, assurance evidence, executive
summaries, and public-facing climate reports. All templates support
multi-format output (Markdown, HTML, JSON, PDF) with SHA-256 provenance
hashing and corporate green colour scheme (#0a4a3a, #167a5b, #1db954).

Templates:
    1.  InterimTargetsSummaryTemplate           - 5/10-year targets by scope with SBTi validation
    2.  AnnualProgressReportTemplate            - SBTi annual disclosure with RAG scoring
    3.  VarianceAnalysisReportTemplate          - LMDI decomposition & Kaya identity analysis
    4.  CorrectiveActionPlanTemplate            - Gap-to-target with MACC curve analysis
    5.  QuarterlyDashboardTemplate              - Board-level 1-page KPI dashboard
    6.  CDPDisclosureTemplate                   - CDP C4.1/C4.2 interim targets disclosure
    7.  TCFDMetricsReportTemplate               - TCFD Metrics & Targets pillar report
    8.  AssuranceEvidencePackageTemplate         - ISO 14064-3 assurance workpapers
    9.  ExecutiveSummaryTemplate                - 1-page C-suite executive summary
    10. PublicDisclosureTemplate                - Public-facing annual climate report

Usage:
    >>> from packs.net_zero.PACK_029_interim_targets.templates import (
    ...     TemplateRegistry,
    ...     InterimTargetsSummaryTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("interim_targets_summary")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("annual_progress_report", data, format="html")

    >>> # Filter by category
    >>> tracking = registry.get_by_category("tracking")

    >>> # Multi-format batch render
    >>> for fmt in ["markdown", "html", "json"]:
    ...     output = registry.render("executive_summary", data, format=fmt)

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

__version__ = "29.0.0"
__pack_id__ = "PACK-029"
__pack_name__ = "Interim Targets Pack"

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .interim_targets_summary import InterimTargetsSummaryTemplate
except ImportError:
    InterimTargetsSummaryTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import InterimTargetsSummaryTemplate")

try:
    from .annual_progress_report import AnnualProgressReportTemplate
except ImportError:
    AnnualProgressReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AnnualProgressReportTemplate")

try:
    from .variance_analysis_report import VarianceAnalysisReportTemplate
except ImportError:
    VarianceAnalysisReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import VarianceAnalysisReportTemplate")

try:
    from .corrective_action_plan import CorrectiveActionPlanTemplate
except ImportError:
    CorrectiveActionPlanTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CorrectiveActionPlanTemplate")

try:
    from .quarterly_dashboard import QuarterlyDashboardTemplate
except ImportError:
    QuarterlyDashboardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import QuarterlyDashboardTemplate")

try:
    from .cdp_disclosure import CDPDisclosureTemplate
except ImportError:
    CDPDisclosureTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CDPDisclosureTemplate")

try:
    from .tcfd_metrics_report import TCFDMetricsReportTemplate
except ImportError:
    TCFDMetricsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TCFDMetricsReportTemplate")

try:
    from .assurance_evidence_package import AssuranceEvidencePackageTemplate
except ImportError:
    AssuranceEvidencePackageTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AssuranceEvidencePackageTemplate")

try:
    from .executive_summary import ExecutiveSummaryTemplate
except ImportError:
    ExecutiveSummaryTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ExecutiveSummaryTemplate")

try:
    from .public_disclosure import PublicDisclosureTemplate
except ImportError:
    PublicDisclosureTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import PublicDisclosureTemplate")


__all__ = [
    # Version info
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Template classes
    "InterimTargetsSummaryTemplate",
    "AnnualProgressReportTemplate",
    "VarianceAnalysisReportTemplate",
    "CorrectiveActionPlanTemplate",
    "QuarterlyDashboardTemplate",
    "CDPDisclosureTemplate",
    "TCFDMetricsReportTemplate",
    "AssuranceEvidencePackageTemplate",
    "ExecutiveSummaryTemplate",
    "PublicDisclosureTemplate",
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
        "name": "interim_targets_summary",
        "class": InterimTargetsSummaryTemplate,
        "description": (
            "Comprehensive interim targets summary rendering 5-year and "
            "10-year interim targets by scope, baseline-to-net-zero pathway "
            "visualization, annual emissions trajectory chart data, cumulative "
            "carbon budget analysis, and SBTi 1.5C validation (42% near-term "
            "check). Includes scope-level target breakdown with progress "
            "tracking and multi-format output."
        ),
        "category": "summary",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "annual_progress_report",
        "class": AnnualProgressReportTemplate,
        "description": (
            "Annual progress report with SBTi annual disclosure format "
            "covering required fields, actual vs target emissions comparison "
            "across all scopes, variance analysis with red/amber/green "
            "performance scoring, initiative deployment status, forward-"
            "looking 3-year projections, year-over-year trends, and "
            "assurance statement section."
        ),
        "category": "tracking",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "variance_analysis_report",
        "class": VarianceAnalysisReportTemplate,
        "description": (
            "LMDI (Logarithmic Mean Divisia Index) decomposition analysis "
            "with Kaya identity waterfall chart data, scope-level and "
            "category-level attribution, root cause classification "
            "(internal initiatives, external factors, M&A, methodology), "
            "year-over-year variance trends, sensitivity analysis, and "
            "corrective action recommendations."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "corrective_action_plan",
        "class": CorrectiveActionPlanTemplate,
        "description": (
            "Gap-to-target corrective action plan with quantified gap "
            "analysis, candidate initiative portfolio sorted by Marginal "
            "Abatement Cost Curve (MACC), phased deployment scheduling "
            "(quick wins, medium-term, strategic), investment requirements "
            "(CapEx/OpEx/NPV/ROI), risk assessment across 5 categories, "
            "and expected reduction impact per initiative."
        ),
        "category": "action",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "quarterly_dashboard",
        "class": QuarterlyDashboardTemplate,
        "description": (
            "Board-level 1-page quarterly KPI dashboard with key emissions "
            "metrics (actual, target, variance, % complete), red/amber/green "
            "milestone alerts, top 5 performing initiatives, top 3 risks "
            "and issues, rolling 4-quarter trend analysis, and next quarter "
            "milestones. Designed for C-suite consumption."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "cdp_disclosure",
        "class": CDPDisclosureTemplate,
        "description": (
            "CDP Climate Change disclosure covering C4.1 (interim targets "
            "description), C4.2 (interim targets table with base year, "
            "target year, reduction %, scope), C4.1a (detailed target "
            "descriptions), cross-references to C5/C6 emissions data, "
            "public disclosure readiness check (10 criteria), and A-list "
            "optimization tips for maximum CDP score."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "tcfd_metrics_report",
        "class": TCFDMetricsReportTemplate,
        "description": (
            "TCFD Metrics and Targets pillar disclosure covering GHG "
            "emissions by scope with interim targets, transition risks and "
            "opportunities linked to targets, forward-looking projected "
            "emissions, scenario analysis integration (1.5C, 2C, STEPS), "
            "carbon pricing impact, internal carbon price application, and "
            "TCFD recommendation alignment scoring."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "assurance_evidence_package",
        "class": AssuranceEvidencePackageTemplate,
        "description": (
            "ISO 14064-3 aligned assurance evidence package with workpaper "
            "structure, 5-tier evidence hierarchy (measured to estimated), "
            "calculation trails (activity data to emission factors to "
            "emissions), variance explanation documentation, data quality "
            "assessment, 15-item assurance provider checklist, materiality "
            "assessment, and control environment review."
        ),
        "category": "assurance",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "executive_summary",
        "class": ExecutiveSummaryTemplate,
        "description": (
            "1-page executive summary for board and C-suite with headline "
            "metrics (total emissions, target, variance, % complete), "
            "performance summary (on-track, ahead, behind), key "
            "achievements with tCO2e impact, key risks, corrective "
            "action recommendations, and next steps with owners and "
            "deadlines."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
    {
        "name": "public_disclosure",
        "class": PublicDisclosureTemplate,
        "description": (
            "Public-facing annual climate report with stakeholder-friendly "
            "language, executive message, climate commitments, emissions "
            "performance, interim targets and progress, key initiatives, "
            "scope breakdown, net-zero pathway visualization data, 12-point "
            "greenwashing compliance check (EU Green Claims Directive), and "
            "links to detailed framework reports (SBTi, CDP, TCFD)."
        ),
        "category": "public",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "29.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-029 Interim Targets Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 interim targets report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON/PDF.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("interim_targets_summary")
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
            "PACK-029 TemplateRegistry initialized with %d templates",
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
            name: Template name (e.g., 'interim_targets_summary').
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
                'summary', 'tracking', 'analysis', 'action',
                'dashboard', 'disclosure', 'assurance', 'executive',
                'public'.

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
            f"TemplateRegistry(pack='PACK-029', templates={self.template_count}, "
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
