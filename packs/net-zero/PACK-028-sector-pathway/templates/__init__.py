# -*- coding: utf-8 -*-
"""
PACK-028 Sector Pathway Pack - Report Templates
====================================================

This package provides 8 report templates for the PACK-028 Sector Pathway
Pack, designed for organisations aligning decarbonisation strategies with
SBTi Sectoral Decarbonization Approach (SDA) pathways, IEA Net Zero by
2050 milestones, and sector-specific intensity convergence benchmarks.
Templates support multi-format output (Markdown, HTML, JSON, PDF) with
SHA-256 provenance hashing and corporate green colour scheme (#0a4a3a,
#167a5b, #1db954).

Templates:
    1.  SectorPathwayReportTemplate           - Sector pathway with SDA/IEA alignment
    2.  IntensityConvergenceReportTemplate     - Intensity metric convergence analysis
    3.  TechnologyRoadmapReportTemplate        - Technology adoption roadmap & CapEx phasing
    4.  AbatementWaterfallReportTemplate       - Abatement waterfall & MACC analysis
    5.  SectorBenchmarkReportTemplate          - Multi-dimensional sector benchmarking
    6.  ScenarioComparisonReportTemplate       - 5-scenario comparison matrix
    7.  SBTiValidationReportTemplate           - SBTi criteria validation & gap analysis
    8.  SectorStrategyReportTemplate           - Board-level sector strategy consolidation

Usage:
    >>> from packs.net_zero.PACK_028_sector_pathway.templates import (
    ...     TemplateRegistry,
    ...     SectorPathwayReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("sector_pathway_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("sbti_validation_report", data, format="html")

    >>> # Filter by category
    >>> pathway_templates = registry.get_by_category("pathway")

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

__version__ = "28.0.0"
__pack_id__ = "PACK-028"
__pack_name__ = "Sector Pathway Pack"

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .sector_pathway_report import SectorPathwayReportTemplate
except ImportError:
    SectorPathwayReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SectorPathwayReportTemplate")

try:
    from .intensity_convergence_report import IntensityConvergenceReportTemplate
except ImportError:
    IntensityConvergenceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import IntensityConvergenceReportTemplate")

try:
    from .technology_roadmap_report import TechnologyRoadmapReportTemplate
except ImportError:
    TechnologyRoadmapReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TechnologyRoadmapReportTemplate")

try:
    from .abatement_waterfall_report import AbatementWaterfallReportTemplate
except ImportError:
    AbatementWaterfallReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AbatementWaterfallReportTemplate")

try:
    from .sector_benchmark_report import SectorBenchmarkReportTemplate
except ImportError:
    SectorBenchmarkReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SectorBenchmarkReportTemplate")

try:
    from .scenario_comparison_report import ScenarioComparisonReportTemplate
except ImportError:
    ScenarioComparisonReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ScenarioComparisonReportTemplate")

try:
    from .sbti_validation_report import SBTiValidationReportTemplate
except ImportError:
    SBTiValidationReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SBTiValidationReportTemplate")

try:
    from .sector_strategy_report import SectorStrategyReportTemplate
except ImportError:
    SectorStrategyReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SectorStrategyReportTemplate")


__all__ = [
    # Version info
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Template classes
    "SectorPathwayReportTemplate",
    "IntensityConvergenceReportTemplate",
    "TechnologyRoadmapReportTemplate",
    "AbatementWaterfallReportTemplate",
    "SectorBenchmarkReportTemplate",
    "ScenarioComparisonReportTemplate",
    "SBTiValidationReportTemplate",
    "SectorStrategyReportTemplate",
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
        "name": "sector_pathway_report",
        "class": SectorPathwayReportTemplate,
        "description": (
            "Sector-specific decarbonisation pathway report aligning with "
            "SBTi Sectoral Decarbonization Approach (SDA) and IEA Net Zero "
            "by 2050. Covers 15+ sectors including power, steel, cement, "
            "aluminium, transport, aviation, shipping, and real estate. "
            "Includes intensity convergence curves (linear, exponential, "
            "S-curve, stepped), year-by-year target tables, activity growth "
            "projections, and gap analysis with confidence intervals."
        ),
        "category": "pathway",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "intensity_convergence_report",
        "class": IntensityConvergenceReportTemplate,
        "description": (
            "Intensity metric convergence analysis tracking sector-specific "
            "intensity metrics (e.g., gCO2/kWh for power, tCO2e/tonne for "
            "steel) against SDA convergence targets. Features CAGR "
            "calculations, projected convergence year estimation, gap "
            "analysis with RAG status, regional breakdown, and year-over-"
            "year trend visualisation with data quality scoring."
        ),
        "category": "convergence",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "technology_roadmap_report",
        "class": TechnologyRoadmapReportTemplate,
        "description": (
            "Technology adoption roadmap for sector decarbonisation covering "
            "Technology Readiness Levels (TRL 1-9), S-curve adoption "
            "modelling, CapEx phasing with NPV analysis, and IEA milestone "
            "tracking. Maps sector-specific technologies (e.g., CCUS, "
            "hydrogen DRI, heat pumps, SAF) with implementation timelines, "
            "abatement potential, and cost trajectories."
        ),
        "category": "technology",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "abatement_waterfall_report",
        "class": AbatementWaterfallReportTemplate,
        "description": (
            "Marginal Abatement Cost Curve (MACC) and waterfall chart "
            "analysis breaking down emissions reduction by lever. Covers "
            "energy efficiency, fuel switching, electrification, CCUS, "
            "process innovation, and demand reduction with cost per tCO2e, "
            "implementation timeline, running totals, cumulative abatement, "
            "and sector-specific lever categorisation."
        ),
        "category": "abatement",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "sector_benchmark_report",
        "class": SectorBenchmarkReportTemplate,
        "description": (
            "Multi-dimensional sector benchmarking across 8 dimensions: "
            "emissions intensity, reduction trajectory, target ambition, "
            "technology readiness, investment commitment, transition risk, "
            "policy alignment, and disclosure quality. Features peer "
            "comparison, sector leader identification, IEA pathway "
            "alignment, percentile rankings, and quartile analysis with "
            "colour-coded scorecards."
        ),
        "category": "benchmarking",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "scenario_comparison_report",
        "class": ScenarioComparisonReportTemplate,
        "description": (
            "Five-scenario comparison matrix (NZE 1.5C, Well Below 2C, "
            "2C, Announced Pledges, Stated Policies) with sector-specific "
            "pathway analysis. Includes investment requirement deltas, "
            "risk-return assessment, technology deployment rates, emissions "
            "trajectory comparison, carbon budget analysis, and optimal "
            "pathway recommendation with scoring methodology."
        ),
        "category": "scenario",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "sbti_validation_report",
        "class": SBTiValidationReportTemplate,
        "description": (
            "SBTi Sectoral Decarbonization Approach validation report "
            "covering 10 near-term criteria (C1-C10), 6 SDA-specific "
            "criteria (SDA-1 to SDA-6), and 5 long-term criteria (LT-1 "
            "to LT-5). Includes pass/fail/partial assessment, gap "
            "identification with remediation priorities, readiness score "
            "calculation, and improvement recommendations for SBTi "
            "target submission readiness."
        ),
        "category": "validation",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
    {
        "name": "sector_strategy_report",
        "class": SectorStrategyReportTemplate,
        "description": (
            "Board-level sector strategy consolidation report combining "
            "pathway analysis, technology roadmap, abatement strategy, "
            "competitive positioning, and investment priorities into a "
            "single executive document. Features CONFIDENTIAL classification, "
            "6 strategic pillars, risk assessment, governance framework, "
            "KPI dashboard, and board recommendations with implementation "
            "roadmap and quarterly milestones."
        ),
        "category": "strategy",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "28.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-028 Sector Pathway Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 sector pathway report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON/PDF.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("sector_pathway_report")
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
            "PACK-028 TemplateRegistry initialized with %d templates",
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
            name: Template name (e.g., 'sector_pathway_report').
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
                'pathway', 'convergence', 'technology', 'abatement',
                'benchmarking', 'scenario', 'validation', 'strategy'.

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
            f"TemplateRegistry(pack='PACK-028', templates={self.template_count}, "
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
