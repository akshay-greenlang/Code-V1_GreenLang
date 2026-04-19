# -*- coding: utf-8 -*-
"""
PACK-046 Intensity Metrics Pack - Report Templates
=========================================================

This package provides 10 report templates for the PACK-046 Intensity Metrics
Pack. Each template supports multiple rendering formats (Markdown, HTML,
JSON, and in some cases XBRL/CSV) with SHA-256 provenance hashing for
audit trail integrity.

Templates:
    1. IntensityExecutiveDashboard      - Top 5 metrics, benchmarks, targets, actions
    2. IntensityDetailedReport          - Full methodology, scope/denominator tables
    3. DecompositionWaterfallReport     - LMDI decomposition waterfall analysis
    4. BenchmarkComparisonReport        - Peer group ranking, gap analysis
    5. TargetPathwayReport              - SBTi pathway vs actual trajectory
    6. ScenarioAnalysisReport           - Monte Carlo, sensitivity, fan charts
    7. UncertaintyReport                - IPCC Tier 1/2 uncertainty bands
    8. ESRSE1IntensityDisclosure        - ESRS E1-6 intensity disclosure
    9. CDPIntensityDisclosure           - CDP C6.10 intensity disclosure
    10. IntensityKPIScorecard           - Traffic-light scorecard per metric

Usage:
    >>> from packs.ghg_accounting.PACK_046_intensity_metrics.templates import (
    ...     TemplateRegistry,
    ...     IntensityExecutiveDashboard,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("intensity_executive_dashboard")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

Author: GreenLang Team
Version: 46.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__pack__ = "PACK-046"
__pack_name__ = "Intensity Metrics Pack"
__templates_count__ = 10

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .intensity_executive_dashboard import IntensityExecutiveDashboard
except ImportError as e:
    logger.warning("Failed to import IntensityExecutiveDashboard: %s", e)
    IntensityExecutiveDashboard = None  # type: ignore[assignment,misc]

try:
    from .intensity_detailed_report import IntensityDetailedReport
except ImportError as e:
    logger.warning("Failed to import IntensityDetailedReport: %s", e)
    IntensityDetailedReport = None  # type: ignore[assignment,misc]

try:
    from .decomposition_waterfall import DecompositionWaterfallReport
except ImportError as e:
    logger.warning("Failed to import DecompositionWaterfallReport: %s", e)
    DecompositionWaterfallReport = None  # type: ignore[assignment,misc]

try:
    from .benchmark_comparison import BenchmarkComparisonReport
except ImportError as e:
    logger.warning("Failed to import BenchmarkComparisonReport: %s", e)
    BenchmarkComparisonReport = None  # type: ignore[assignment,misc]

try:
    from .target_pathway_report import TargetPathwayReport
except ImportError as e:
    logger.warning("Failed to import TargetPathwayReport: %s", e)
    TargetPathwayReport = None  # type: ignore[assignment,misc]

try:
    from .scenario_analysis_report import ScenarioAnalysisReport
except ImportError as e:
    logger.warning("Failed to import ScenarioAnalysisReport: %s", e)
    ScenarioAnalysisReport = None  # type: ignore[assignment,misc]

try:
    from .uncertainty_report import UncertaintyReport
except ImportError as e:
    logger.warning("Failed to import UncertaintyReport: %s", e)
    UncertaintyReport = None  # type: ignore[assignment,misc]

try:
    from .esrs_e1_intensity_disclosure import ESRSE1IntensityDisclosure
except ImportError as e:
    logger.warning("Failed to import ESRSE1IntensityDisclosure: %s", e)
    ESRSE1IntensityDisclosure = None  # type: ignore[assignment,misc]

try:
    from .cdp_intensity_disclosure import CDPIntensityDisclosure
except ImportError as e:
    logger.warning("Failed to import CDPIntensityDisclosure: %s", e)
    CDPIntensityDisclosure = None  # type: ignore[assignment,misc]

try:
    from .intensity_kpi_scorecard import IntensityKPIScorecard
except ImportError as e:
    logger.warning("Failed to import IntensityKPIScorecard: %s", e)
    IntensityKPIScorecard = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "IntensityExecutiveDashboard",
    "IntensityDetailedReport",
    "DecompositionWaterfallReport",
    "BenchmarkComparisonReport",
    "TargetPathwayReport",
    "ScenarioAnalysisReport",
    "UncertaintyReport",
    "ESRSE1IntensityDisclosure",
    "CDPIntensityDisclosure",
    "IntensityKPIScorecard",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
    # Type aliases
    "TemplateClass",
    "TemplateInstance",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TemplateClass = Union[
    Type[IntensityExecutiveDashboard],
    Type[IntensityDetailedReport],
    Type[DecompositionWaterfallReport],
    Type[BenchmarkComparisonReport],
    Type[TargetPathwayReport],
    Type[ScenarioAnalysisReport],
    Type[UncertaintyReport],
    Type[ESRSE1IntensityDisclosure],
    Type[CDPIntensityDisclosure],
    Type[IntensityKPIScorecard],
]

TemplateInstance = Union[
    IntensityExecutiveDashboard,
    IntensityDetailedReport,
    DecompositionWaterfallReport,
    BenchmarkComparisonReport,
    TargetPathwayReport,
    ScenarioAnalysisReport,
    UncertaintyReport,
    ESRSE1IntensityDisclosure,
    CDPIntensityDisclosure,
    IntensityKPIScorecard,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "intensity_executive_dashboard",
        "class": IntensityExecutiveDashboard,
        "description": (
            "Executive dashboard with top 5 intensity metrics, YoY change "
            "arrows, benchmark percentile positioning, target progress "
            "indicators, decomposition highlights, and action items."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "intensity_detailed_report",
        "class": IntensityDetailedReport,
        "description": (
            "Comprehensive intensity report with methodology description, "
            "scope configuration, denominator details, intensity-by-scope "
            "and intensity-by-denominator tables, multi-year time series, "
            "entity breakdown, data sources, and limitations."
        ),
        "category": "detailed",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "decomposition_waterfall",
        "class": DecompositionWaterfallReport,
        "description": (
            "LMDI decomposition waterfall report showing activity, structure, "
            "and intensity effects with entity contributions, waterfall chart "
            "data, narrative interpretation, and closure validation."
        ),
        "category": "decomposition",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "benchmark_comparison",
        "class": BenchmarkComparisonReport,
        "description": (
            "Benchmark comparison report with peer group definitions, "
            "normalisation methodology, ranking tables, percentile chart "
            "data, gap analysis, sector distributions, and improvement "
            "recommendations."
        ),
        "category": "benchmark",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "target_pathway_report",
        "class": TargetPathwayReport,
        "description": (
            "Target pathway report with SBTi methodology, base year summary, "
            "pathway chart data (actual vs target), annual progress table, "
            "gap analysis, required reduction rate, and trajectory projection."
        ),
        "category": "targets",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "scenario_analysis_report",
        "class": ScenarioAnalysisReport,
        "description": (
            "Scenario analysis report with scenario definitions, base case, "
            "scenario results table, Monte Carlo distribution data, "
            "probability of target achievement, sensitivity tornado chart "
            "data, key drivers, and recommendations."
        ),
        "category": "scenario",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "uncertainty_report",
        "class": UncertaintyReport,
        "description": (
            "Uncertainty analysis report with IPCC Tier 1/2 methodology, "
            "data quality summary table, uncertainty by metric, combined "
            "uncertainty bands, confidence intervals, data improvement "
            "recommendations, and quality trend."
        ),
        "category": "uncertainty",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "esrs_e1_intensity_disclosure",
        "class": ESRSE1IntensityDisclosure,
        "description": (
            "ESRS E1-6 intensity disclosure with GHG intensity per net "
            "revenue (mandatory), sector physical intensity metrics, "
            "methodology description, data quality statement, and "
            "comparative prior year figures. Supports XBRL tagging."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "pdf", "json", "xbrl"],
        "version": "46.0.0",
    },
    {
        "name": "cdp_intensity_disclosure",
        "class": CDPIntensityDisclosure,
        "description": (
            "CDP C6.10 intensity disclosure with Scope 1+2 intensity "
            "figures, metric denominators, percentage change from prior "
            "year, direction and reason for change, and sector-specific "
            "module fields."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
    {
        "name": "intensity_kpi_scorecard",
        "class": IntensityKPIScorecard,
        "description": (
            "Traffic-light KPI scorecard with per-metric cards showing "
            "current value, target, status (on-target/at-risk/off-target), "
            "trend indicators, action items, and next review date."
        ),
        "category": "kpi",
        "formats": ["markdown", "html", "pdf", "json"],
        "version": "46.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-046 Intensity Metrics report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 report templates. Templates can be listed, filtered by
    category, retrieved by name, and rendered in markdown/HTML/JSON
    formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("intensity_executive_dashboard")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, TemplateInstance] = {}

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
        return [
            defn["name"] for defn in TEMPLATE_CATALOG
            if defn["class"] is not None
        ]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TemplateInstance:
        """
        Get a template instance by name.

        Args:
            name: Template name (e.g., 'intensity_executive_dashboard').
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
            category: Category string (e.g., 'executive', 'detailed',
                      'decomposition', 'benchmark', 'targets', 'scenario',
                      'uncertainty', 'regulatory', 'kpi').

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
