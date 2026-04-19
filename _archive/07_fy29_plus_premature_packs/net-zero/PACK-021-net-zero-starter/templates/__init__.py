# -*- coding: utf-8 -*-
"""
PACK-021 Net Zero Starter Pack - Report Templates
=====================================================

This package provides 8 report templates for the PACK-021 Net Zero Starter
Pack, covering all key aspects of net-zero strategy development, baseline
assessment, target validation, reduction planning, offset management,
maturity scoring, progress tracking, and peer benchmarking. Each template
supports three rendering formats: Markdown, HTML (with inline CSS), and JSON.
All templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. NetZeroStrategyReportTemplate     - Executive net-zero strategy document
    2. GHGBaselineReportTemplate         - Detailed GHG inventory baseline
    3. TargetValidationReportTemplate    - SBTi target validation & compliance
    4. ReductionRoadmapReportTemplate    - Phased reduction roadmap with MACC
    5. OffsetPortfolioReportTemplate     - Carbon credit portfolio & quality
    6. NetZeroScorecardReportTemplate    - Net-zero maturity scorecard (8 dimensions)
    7. ProgressDashboardReportTemplate   - Progress tracking dashboard with RAG
    8. BenchmarkComparisonReportTemplate - Peer benchmarking comparison

Usage:
    >>> from packs.net_zero.PACK_021_net_zero_starter.templates import (
    ...     TemplateRegistry,
    ...     NetZeroStrategyReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("net_zero_strategy_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("ghg_baseline_report", data, format="html")

    >>> # Filter by category
    >>> baseline_templates = registry.get_by_category("baseline")

Author: GreenLang Team
Version: 21.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .net_zero_strategy_report import (
        NetZeroStrategyReportTemplate,
    )
except ImportError:
    NetZeroStrategyReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import NetZeroStrategyReportTemplate")

try:
    from .ghg_baseline_report import (
        GHGBaselineReportTemplate,
    )
except ImportError:
    GHGBaselineReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GHGBaselineReportTemplate")

try:
    from .target_validation_report import (
        TargetValidationReportTemplate,
    )
except ImportError:
    TargetValidationReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TargetValidationReportTemplate")

try:
    from .reduction_roadmap_report import (
        ReductionRoadmapReportTemplate,
    )
except ImportError:
    ReductionRoadmapReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ReductionRoadmapReportTemplate")

try:
    from .offset_portfolio_report import (
        OffsetPortfolioReportTemplate,
    )
except ImportError:
    OffsetPortfolioReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import OffsetPortfolioReportTemplate")

try:
    from .net_zero_scorecard_report import (
        NetZeroScorecardReportTemplate,
    )
except ImportError:
    NetZeroScorecardReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import NetZeroScorecardReportTemplate")

try:
    from .progress_dashboard_report import (
        ProgressDashboardReportTemplate,
    )
except ImportError:
    ProgressDashboardReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ProgressDashboardReportTemplate")

try:
    from .benchmark_comparison_report import (
        BenchmarkComparisonReportTemplate,
    )
except ImportError:
    BenchmarkComparisonReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import BenchmarkComparisonReportTemplate")


__all__ = [
    # Template classes
    "NetZeroStrategyReportTemplate",
    "GHGBaselineReportTemplate",
    "TargetValidationReportTemplate",
    "ReductionRoadmapReportTemplate",
    "OffsetPortfolioReportTemplate",
    "NetZeroScorecardReportTemplate",
    "ProgressDashboardReportTemplate",
    "BenchmarkComparisonReportTemplate",
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
        "name": "net_zero_strategy_report",
        "class": NetZeroStrategyReportTemplate,
        "description": (
            "Executive-level net-zero strategy document with organization profile, "
            "GHG baseline summary, scope split, near-term and long-term targets, "
            "reduction pathway, offset strategy, implementation timeline, investment "
            "requirements, risk assessment, and governance framework."
        ),
        "category": "strategy",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "ghg_baseline_report",
        "class": GHGBaselineReportTemplate,
        "description": (
            "Detailed GHG inventory baseline report with methodology documentation, "
            "organizational boundary, Scope 1 source-type breakdown, Scope 2 "
            "location vs market comparison, Scope 3 all 15 categories with "
            "relevance assessment, data quality matrix, base year statement, "
            "and emission factors used."
        ),
        "category": "baseline",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "target_validation_report",
        "class": TargetValidationReportTemplate,
        "description": (
            "SBTi target validation and compliance report with 10-point criteria "
            "checklist (pass/fail), pathway details with yearly projections, "
            "scope coverage analysis, 1.5C ambition assessment, milestones "
            "timeline, sector benchmark comparison, and recommendations."
        ),
        "category": "targets",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "reduction_roadmap_report",
        "class": ReductionRoadmapReportTemplate,
        "description": (
            "Phased reduction roadmap with emissions hotspot analysis, abatement "
            "options summary, Marginal Abatement Cost Curve data sorted by cost, "
            "short/medium/long-term phased roadmap, CapEx/OpEx investment summary, "
            "cumulative impact projection, quick wins with payback under 2 years, "
            "and implementation dependencies."
        ),
        "category": "reduction",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "offset_portfolio_report",
        "class": OffsetPortfolioReportTemplate,
        "description": (
            "Carbon credit portfolio quality report with residual emissions budget, "
            "portfolio composition by credit type, 6-dimension quality assessment "
            "(additionality, permanence, leakage, MRV, co-benefits, registry), "
            "SBTi neutralization vs compensation compliance, VCMI claims alignment, "
            "credit retirement schedule, cost projection, and recommendations."
        ),
        "category": "offsets",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "net_zero_scorecard_report",
        "class": NetZeroScorecardReportTemplate,
        "description": (
            "Net-zero maturity scorecard across 8 dimensions (governance, baseline, "
            "targets, reduction, data quality, engagement, offsets, disclosure) "
            "with overall score, maturity level badges, dimension detail cards "
            "with progress bars, radar chart data, priority recommendations, "
            "improvement roadmap, and peer comparison context."
        ),
        "category": "scorecard",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "progress_dashboard_report",
        "class": ProgressDashboardReportTemplate,
        "description": (
            "Progress tracking dashboard with KPI metric cards and YoY change, "
            "year-over-year emissions trend, actual vs pathway target table, "
            "scope breakdown trend, intensity metrics trend, action implementation "
            "status with completion bars, gap analysis with RAG indicators, "
            "corrective actions required, and forecast projections."
        ),
        "category": "progress",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
    {
        "name": "benchmark_comparison_report",
        "class": BenchmarkComparisonReportTemplate,
        "description": (
            "Peer benchmarking comparison with company profile, sector context, "
            "KPI comparison table (your value vs sector average vs leader), "
            "percentile rankings with quartile bars, identified strengths and "
            "gaps, best practice insights from leaders, and prioritized "
            "improvement opportunities."
        ),
        "category": "benchmark",
        "formats": ["markdown", "html", "json"],
        "version": "21.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-021 Net Zero Starter Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 net-zero report templates. Templates can be listed, filtered by
    category, retrieved by name, and rendered in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("net_zero_strategy_report")
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
            name: Template name (e.g., 'net_zero_strategy_report').
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
            category: Category string. Valid categories:
                'strategy', 'baseline', 'targets', 'reduction',
                'offsets', 'scorecard', 'progress', 'benchmark'.

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
            f"TemplateRegistry(templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )
