# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Report Templates
=========================================================

This package provides 8 quick-wins identification report templates for the
PACK-033 Quick Wins Identifier Pack. Each template supports three
rendering formats: Markdown, HTML (with inline CSS), and JSON. All
templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. QuickWinsScanReportTemplate           - Facility scan results report
    2. PrioritizedActionsReportTemplate      - Ranked actions with MCDA scores
    3. PaybackAnalysisReportTemplate         - Financial analysis of quick wins
    4. CarbonReductionReportTemplate         - Emissions reduction report
    5. ImplementationPlanReportTemplate      - Implementation roadmap
    6. ProgressDashboardTemplate             - Progress tracking dashboard
    7. ExecutiveSummaryReportTemplate        - C-suite summary
    8. RebateOpportunitiesReportTemplate    - Utility rebate report

Usage:
    >>> from packs.energy_efficiency.PACK_033_quick_wins_identifier.templates import (
    ...     TemplateRegistry,
    ...     QuickWinsScanReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("quick_wins_scan_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 33.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .quick_wins_scan_report import QuickWinsScanReportTemplate
from .prioritized_actions_report import PrioritizedActionsReportTemplate
from .payback_analysis_report import PaybackAnalysisReportTemplate
from .carbon_reduction_report import CarbonReductionReportTemplate
from .implementation_plan_report import ImplementationPlanReportTemplate
from .progress_dashboard import ProgressDashboardTemplate
from .executive_summary_report import ExecutiveSummaryReportTemplate
from .rebate_opportunities_report import RebateOpportunitiesReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "QuickWinsScanReportTemplate",
    "PrioritizedActionsReportTemplate",
    "PaybackAnalysisReportTemplate",
    "CarbonReductionReportTemplate",
    "ImplementationPlanReportTemplate",
    "ProgressDashboardTemplate",
    "ExecutiveSummaryReportTemplate",
    "RebateOpportunitiesReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[QuickWinsScanReportTemplate],
    Type[PrioritizedActionsReportTemplate],
    Type[PaybackAnalysisReportTemplate],
    Type[CarbonReductionReportTemplate],
    Type[ImplementationPlanReportTemplate],
    Type[ProgressDashboardTemplate],
    Type[ExecutiveSummaryReportTemplate],
    Type[RebateOpportunitiesReportTemplate],
]

TemplateInstance = Union[
    QuickWinsScanReportTemplate,
    PrioritizedActionsReportTemplate,
    PaybackAnalysisReportTemplate,
    CarbonReductionReportTemplate,
    ImplementationPlanReportTemplate,
    ProgressDashboardTemplate,
    ExecutiveSummaryReportTemplate,
    RebateOpportunitiesReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "quick_wins_scan_report",
        "class": QuickWinsScanReportTemplate,
        "description": (
            "Facility quick-wins scan results report with executive summary, "
            "facility profile, scan methodology, categorized quick wins with "
            "savings estimates, payback periods, priority rankings, category "
            "breakdown with pie chart data, and next steps."
        ),
        "category": "scan",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "prioritized_actions_report",
        "class": PrioritizedActionsReportTemplate,
        "description": (
            "Prioritized actions report with Multi-Criteria Decision Analysis "
            "(MCDA) scoring, ranking summary, top 10 actions table, criteria "
            "weight breakdown, Pareto frontier analysis, implementation phases, "
            "and inter-action dependency mapping."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "payback_analysis_report",
        "class": PaybackAnalysisReportTemplate,
        "description": (
            "Financial payback analysis report with portfolio summary, "
            "measure-by-measure analysis including NPV, IRR, payback period, "
            "and ROI, cash flow projections, sensitivity analysis with "
            "scenario modeling, and investment recommendations."
        ),
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "carbon_reduction_report",
        "class": CarbonReductionReportTemplate,
        "description": (
            "Carbon emissions reduction report with carbon summary, scope "
            "breakdown (1/2/3), measure-level reductions with marginal "
            "abatement costs, cumulative projections, SBTi alignment "
            "assessment, and location vs market-based comparison."
        ),
        "category": "carbon",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "implementation_plan_report",
        "class": ImplementationPlanReportTemplate,
        "description": (
            "Implementation roadmap report with plan overview, phased "
            "timeline with Gantt-style data, resource requirements, "
            "budget breakdown by category, rebate opportunities, risk "
            "mitigation strategies, and milestone tracking."
        ),
        "category": "planning",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "progress_dashboard",
        "class": ProgressDashboardTemplate,
        "description": (
            "Progress tracking dashboard with KPI cards (total savings, "
            "ROI, completion rate, CO2e avoided, energy saved), monthly "
            "savings trend, implementation status by measure, variance "
            "analysis with root causes, and active alerts."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "C-suite executive summary with key financial and environmental "
            "KPIs, financial impact analysis (NPV, IRR, BCR), environmental "
            "impact metrics, strategic recommendations, risk summary, and "
            "top 5 quick-win highlights for board-level presentation."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
    {
        "name": "rebate_opportunities_report",
        "class": RebateOpportunitiesReportTemplate,
        "description": (
            "Utility and government rebate opportunities report with rebate "
            "summary, program matching by utility/measure, application status "
            "tracking, timeline management, net cost impact analysis with "
            "effective payback recalculation, and rebate stacking analysis."
        ),
        "category": "rebate",
        "formats": ["markdown", "html", "json"],
        "version": "33.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-033 Quick Wins Identifier report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 quick-wins report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("quick_wins_scan_report")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, TemplateInstance] = {}

        for defn in TEMPLATE_CATALOG:
            self._templates[defn["name"]] = defn

        logger.info(
            "TemplateRegistry initialized with %d templates",
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
        ]

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List of template name strings.
        """
        return [defn["name"] for defn in TEMPLATE_CATALOG]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TemplateInstance:
        """
        Get a template instance by name.

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'quick_wins_scan_report').
            config: Optional configuration overrides.

        Returns:
            Template instance with render_markdown, render_html, render_json.

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
    ) -> TemplateInstance:
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
            format: Output format ('markdown', 'html', 'json').
            config: Optional template configuration.

        Returns:
            Rendered content (str for markdown/html, dict for json).

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
        else:
            raise ValueError(
                f"Unsupported format '{format}'. Use 'markdown', 'html', or 'json'."
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
            category: Category string (e.g., 'scan', 'analysis',
                      'financial', 'carbon', 'planning', 'dashboard',
                      'executive', 'rebate').

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
            if defn["category"] == category
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
            f"TemplateRegistry(templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )
