# -*- coding: utf-8 -*-
"""
PACK-026 SME Net Zero Pack - Report Templates
=================================================

This package provides 8 report templates for the PACK-026 SME Net Zero
Pack, optimised for small and medium enterprise simplicity with visual
dashboards, 1-2 page layouts, and multi-format output (Markdown, HTML,
JSON, Excel). All templates include SHA-256 provenance hashing and
green colour scheme (#1b5e20, #2e7d32).

Templates:
    1. SMEBaselineReportTemplate          - 1-2 page visual emissions baseline
    2. SMEQuickWinsReportTemplate         - Top 5-10 actions ranked by ROI
    3. SMEGrantReportTemplate             - Matched grants with eligibility scores
    4. SMEBoardBriefTemplate              - 1-page board/leadership executive brief
    5. SMERoadmapReportTemplate           - 3-year decarbonization roadmap
    6. SMEProgressDashboardTemplate       - Annual KPI tracking dashboard
    7. SMECertificationSubmissionTemplate - Certification submission documents
    8. SMEAccountingGuideTemplate         - Accounting software integration guide

Usage:
    >>> from packs.net_zero.PACK_026_sme_net_zero.templates import (
    ...     TemplateRegistry,
    ...     SMEBaselineReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("sme_baseline_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("sme_quick_wins_report", data, format="html")

    >>> # Filter by category
    >>> baseline_templates = registry.get_by_category("baseline")

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

__version__ = "26.0.0"
__pack_id__ = "PACK-026"
__pack_name__ = "SME Net Zero Pack"

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .sme_baseline_report import SMEBaselineReportTemplate
except ImportError:
    SMEBaselineReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMEBaselineReportTemplate")

try:
    from .sme_quick_wins_report import SMEQuickWinsReportTemplate
except ImportError:
    SMEQuickWinsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMEQuickWinsReportTemplate")

try:
    from .sme_grant_report import SMEGrantReportTemplate
except ImportError:
    SMEGrantReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMEGrantReportTemplate")

try:
    from .sme_board_brief import SMEBoardBriefTemplate
except ImportError:
    SMEBoardBriefTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMEBoardBriefTemplate")

try:
    from .sme_roadmap_report import SMERoadmapReportTemplate
except ImportError:
    SMERoadmapReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMERoadmapReportTemplate")

try:
    from .sme_progress_dashboard import SMEProgressDashboardTemplate
except ImportError:
    SMEProgressDashboardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMEProgressDashboardTemplate")

try:
    from .sme_certification_submission import SMECertificationSubmissionTemplate
except ImportError:
    SMECertificationSubmissionTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMECertificationSubmissionTemplate")

try:
    from .sme_accounting_guide import SMEAccountingGuideTemplate
except ImportError:
    SMEAccountingGuideTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SMEAccountingGuideTemplate")


__all__ = [
    # Version info
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Template classes
    "SMEBaselineReportTemplate",
    "SMEQuickWinsReportTemplate",
    "SMEGrantReportTemplate",
    "SMEBoardBriefTemplate",
    "SMERoadmapReportTemplate",
    "SMEProgressDashboardTemplate",
    "SMECertificationSubmissionTemplate",
    "SMEAccountingGuideTemplate",
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
        "name": "sme_baseline_report",
        "class": SMEBaselineReportTemplate,
        "description": (
            "1-2 page visual emissions baseline dashboard with executive summary, "
            "scope 1/2/3 breakdown with bar charts, industry peer comparison with "
            "percentile ranking, data quality score (Bronze/Silver/Gold), top 3 "
            "emission sources, and recommended next steps."
        ),
        "category": "baseline",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "26.0.0",
    },
    {
        "name": "sme_quick_wins_report",
        "class": SMEQuickWinsReportTemplate,
        "description": (
            "Top 5-10 quick wins ranked by ROI showing emissions reduction, "
            "implementation cost, annual savings, payback period, IRR, difficulty "
            "rating (1-5), prerequisites, 6-24 month Gantt timeline, and 5-year "
            "total investment/savings summary."
        ),
        "category": "actions",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "26.0.0",
    },
    {
        "name": "sme_grant_report",
        "class": SMEGrantReportTemplate,
        "description": (
            "Matched grants summary with top 3-5 grants showing funding body, "
            "funding range, eligibility score (0-100), deadline, requirements "
            "checklist, documentation needed, application calendar, and pre-filled "
            "data fields for applications."
        ),
        "category": "funding",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "26.0.0",
    },
    {
        "name": "sme_board_brief",
        "class": SMEBoardBriefTemplate,
        "description": (
            "1-page executive summary for board/leadership with current state "
            "(emissions baseline, intensity, peer comparison), 2030/2050 targets, "
            "top 3 quick wins with ROI, grant opportunities, investment vs returns, "
            "risk/opportunity summary, and recommended decision (approve/reject)."
        ),
        "category": "governance",
        "formats": ["markdown", "html", "json"],
        "version": "26.0.0",
    },
    {
        "name": "sme_roadmap_report",
        "class": SMERoadmapReportTemplate,
        "description": (
            "3-year decarbonization roadmap with Year 1 quick wins (LED, renewable "
            "PPA, waste), Year 2 strategic actions (HVAC, EV fleet, supplier), "
            "Year 3 long-term investments (solar PV, heat pumps, process), MACC "
            "curve, emissions trajectory, budget allocation, and milestones/KPIs."
        ),
        "category": "roadmap",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "26.0.0",
    },
    {
        "name": "sme_progress_dashboard",
        "class": SMEProgressDashboardTemplate,
        "description": (
            "Annual KPI tracking dashboard with year-over-year comparison, progress "
            "bars to 2030/2050 targets, scope 1/2/3 trends (last 3 years), quick wins "
            "implementation status, grant funding tracker, cost savings (realized vs "
            "projected), and next quarter actions."
        ),
        "category": "progress",
        "formats": ["markdown", "html", "json"],
        "version": "26.0.0",
    },
    {
        "name": "sme_certification_submission",
        "class": SMECertificationSubmissionTemplate,
        "description": (
            "Pre-filled certification submission documents for SME Climate Hub "
            "(commitment letter, baseline, action plan), B Corp Climate Collective "
            "(impact assessment), ISO 14001 (EMS readiness), and Carbon Trust "
            "(footprint statement, reduction plan) with evidence links and checklists."
        ),
        "category": "certification",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "26.0.0",
    },
    {
        "name": "sme_accounting_guide",
        "class": SMEAccountingGuideTemplate,
        "description": (
            "Step-by-step guide for connecting Xero/QuickBooks/Sage accounting "
            "software with GL account code to Scope 3 category mappings, carbon "
            "cost allocation methodology, monthly P&L carbon tracking, and tax "
            "deduction guidance for energy efficiency capital allowances."
        ),
        "category": "integration",
        "formats": ["markdown", "html", "json"],
        "version": "26.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-026 SME Net Zero Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 SME net-zero report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON/Excel.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("sme_baseline_report")
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
            "PACK-026 TemplateRegistry initialized with %d templates",
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
            name: Template name (e.g., 'sme_baseline_report').
            config: Optional configuration overrides.

        Returns:
            Template instance with render_markdown, render_html,
            render_json, and optionally render_excel methods.

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
            format: Output format ('markdown', 'html', 'json', 'excel').
            config: Optional template configuration.

        Returns:
            Rendered content (str for markdown/html, dict for json/excel).

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
        elif format == "excel":
            if hasattr(template, "render_excel"):
                return template.render_excel(data)
            raise ValueError(
                f"Template '{template_name}' does not support Excel format."
            )
        else:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Use 'markdown', 'html', 'json', or 'excel'."
            )

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get metadata about a specific template.

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

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string. Valid categories:
                'baseline', 'actions', 'funding', 'governance',
                'roadmap', 'progress', 'certification', 'integration'.

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

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return (
            f"TemplateRegistry(pack='PACK-026', templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )
