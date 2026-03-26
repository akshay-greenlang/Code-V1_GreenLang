# -*- coding: utf-8 -*-
"""
PACK-047 GHG Emissions Benchmark Pack - Report Templates
=============================================================

This package provides 10 report templates for the PACK-047 GHG Emissions
Benchmark Pack. Each template supports multiple rendering formats (Markdown,
HTML, JSON, and in some cases XBRL) with SHA-256 provenance hashing for
audit trail integrity.

Templates:
    1. BenchmarkExecutiveDashboard  - Top-line benchmark position, ITR, peer rank
    2. PeerComparisonReport         - Full peer group ranking and gap analysis
    3. PathwayAlignmentReport       - Pathway alignment scores and deviation analysis
    4. TrajectoryAnalysisReport     - CARR ranking, convergence trends, momentum
    5. PortfolioBenchmarkReport     - WACI, carbon footprint, PCAF quality summary
    6. TransitionRiskReport         - Transition risk heat map and financial exposure
    7. DataQualityReport            - PCAF quality ladder scores by dimension
    8. ESRSE1BenchmarkDisclosure    - ESRS E1 benchmark comparison disclosure + XBRL
    9. CDPClimateBenchmarkSection   - CDP C6/C7 benchmark context and band positioning
    10. SFDRPAIBenchmarkReport      - SFDR PAI 1-3 benchmark and Article 8/9 compliance

Usage:
    >>> from packs.ghg_accounting.PACK_047_benchmark.templates import (
    ...     TemplateRegistry,
    ...     BenchmarkExecutiveDashboard,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("benchmark_executive_dashboard")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

Author: GreenLang Team
Version: 47.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__pack__ = "PACK-047"
__pack_name__ = "GHG Emissions Benchmark Pack"
__templates_count__ = 10

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .benchmark_executive_dashboard import BenchmarkExecutiveDashboard
except ImportError as e:
    logger.warning("Failed to import BenchmarkExecutiveDashboard: %s", e)
    BenchmarkExecutiveDashboard = None  # type: ignore[assignment,misc]

try:
    from .peer_comparison_report import PeerComparisonReport
except ImportError as e:
    logger.warning("Failed to import PeerComparisonReport: %s", e)
    PeerComparisonReport = None  # type: ignore[assignment,misc]

try:
    from .pathway_alignment_report import PathwayAlignmentReport
except ImportError as e:
    logger.warning("Failed to import PathwayAlignmentReport: %s", e)
    PathwayAlignmentReport = None  # type: ignore[assignment,misc]

try:
    from .trajectory_analysis_report import TrajectoryAnalysisReport
except ImportError as e:
    logger.warning("Failed to import TrajectoryAnalysisReport: %s", e)
    TrajectoryAnalysisReport = None  # type: ignore[assignment,misc]

try:
    from .portfolio_benchmark_report import PortfolioBenchmarkReport
except ImportError as e:
    logger.warning("Failed to import PortfolioBenchmarkReport: %s", e)
    PortfolioBenchmarkReport = None  # type: ignore[assignment,misc]

try:
    from .transition_risk_report import TransitionRiskReport
except ImportError as e:
    logger.warning("Failed to import TransitionRiskReport: %s", e)
    TransitionRiskReport = None  # type: ignore[assignment,misc]

try:
    from .data_quality_report import DataQualityReport
except ImportError as e:
    logger.warning("Failed to import DataQualityReport: %s", e)
    DataQualityReport = None  # type: ignore[assignment,misc]

try:
    from .esrs_e1_benchmark_disclosure import ESRSE1BenchmarkDisclosure
except ImportError as e:
    logger.warning("Failed to import ESRSE1BenchmarkDisclosure: %s", e)
    ESRSE1BenchmarkDisclosure = None  # type: ignore[assignment,misc]

try:
    from .cdp_climate_benchmark_section import CDPClimateBenchmarkSection
except ImportError as e:
    logger.warning("Failed to import CDPClimateBenchmarkSection: %s", e)
    CDPClimateBenchmarkSection = None  # type: ignore[assignment,misc]

try:
    from .sfdr_pai_benchmark_report import SFDRPAIBenchmarkReport
except ImportError as e:
    logger.warning("Failed to import SFDRPAIBenchmarkReport: %s", e)
    SFDRPAIBenchmarkReport = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "BenchmarkExecutiveDashboard",
    "PeerComparisonReport",
    "PathwayAlignmentReport",
    "TrajectoryAnalysisReport",
    "PortfolioBenchmarkReport",
    "TransitionRiskReport",
    "DataQualityReport",
    "ESRSE1BenchmarkDisclosure",
    "CDPClimateBenchmarkSection",
    "SFDRPAIBenchmarkReport",
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
    Type[BenchmarkExecutiveDashboard],
    Type[PeerComparisonReport],
    Type[PathwayAlignmentReport],
    Type[TrajectoryAnalysisReport],
    Type[PortfolioBenchmarkReport],
    Type[TransitionRiskReport],
    Type[DataQualityReport],
    Type[ESRSE1BenchmarkDisclosure],
    Type[CDPClimateBenchmarkSection],
    Type[SFDRPAIBenchmarkReport],
]

TemplateInstance = Union[
    BenchmarkExecutiveDashboard,
    PeerComparisonReport,
    PathwayAlignmentReport,
    TrajectoryAnalysisReport,
    PortfolioBenchmarkReport,
    TransitionRiskReport,
    DataQualityReport,
    ESRSE1BenchmarkDisclosure,
    CDPClimateBenchmarkSection,
    SFDRPAIBenchmarkReport,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "benchmark_executive_dashboard",
        "class": BenchmarkExecutiveDashboard,
        "description": (
            "Executive dashboard with top-line benchmark position, implied "
            "temperature rise score, peer group percentile, pathway alignment "
            "status, transition risk heat map, and key action items."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "peer_comparison_report",
        "class": PeerComparisonReport,
        "description": (
            "Comprehensive peer comparison report with peer group definition, "
            "normalisation methodology, ranking tables, percentile chart data, "
            "gap analysis, sector distributions, and improvement actions."
        ),
        "category": "peer",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "pathway_alignment_report",
        "class": PathwayAlignmentReport,
        "description": (
            "Pathway alignment report with IEA NZE, IPCC AR6, SBTi SDA, "
            "OECM, and TPI pathway scores, deviation analysis, sector "
            "convergence charts, and alignment improvement recommendations."
        ),
        "category": "pathway",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "trajectory_analysis_report",
        "class": TrajectoryAnalysisReport,
        "description": (
            "Trajectory analysis report with CARR ranking, convergence trends, "
            "momentum indicators, structural break detection, and fan chart "
            "distribution envelope for forward-looking emissions trajectories."
        ),
        "category": "trajectory",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "portfolio_benchmark_report",
        "class": PortfolioBenchmarkReport,
        "description": (
            "Portfolio benchmark report with WACI comparison, carbon footprint "
            "trends, sector attribution waterfall, top 10 contributors, PCAF "
            "distribution, and holdings heatmap for investor portfolios."
        ),
        "category": "portfolio",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "transition_risk_report",
        "class": TransitionRiskReport,
        "description": (
            "Transition risk report with composite risk score, ITR gauge, "
            "stranding risk timeline, regulatory exposure (EU ETS/CBAM), "
            "competitive positioning radar, and carbon price sensitivity."
        ),
        "category": "risk",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "data_quality_report",
        "class": DataQualityReport,
        "description": (
            "Data quality report with quality histogram, coverage analysis, "
            "source hierarchy breakdown, confidence intervals, improvement "
            "recommendations, and PCAF quality score distribution."
        ),
        "category": "quality",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "esrs_e1_benchmark_disclosure",
        "class": ESRSE1BenchmarkDisclosure,
        "description": (
            "ESRS E1-4 benchmark comparison disclosure with sector-level "
            "GHG benchmark context paragraphs, EU Taxonomy alignment, XBRL "
            "tag mapping, and cross-reference to PACK-046 intensity metrics."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json", "xbrl"],
        "version": "47.0.0",
    },
    {
        "name": "cdp_climate_benchmark_section",
        "class": CDPClimateBenchmarkSection,
        "description": (
            "CDP C6/C7 benchmark context section with performance band "
            "positioning (A through D-), sector supplementary data, "
            "emissions benchmark metrics, and sector comparison analysis."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
    {
        "name": "sfdr_pai_benchmark_report",
        "class": SFDRPAIBenchmarkReport,
        "description": (
            "SFDR PAI benchmark report with PAI indicators 1-3 (GHG emissions, "
            "carbon footprint, GHG intensity), portfolio vs index comparison, "
            "Article 8/9 compliance assessment, and EU Taxonomy benchmark."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "47.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-047 GHG Emissions Benchmark report templates.

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
        >>> template = registry.get("benchmark_executive_dashboard")
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
            name: Template name (e.g., 'benchmark_executive_dashboard').
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
            category: Category string (e.g., 'executive', 'peer',
                      'pathway', 'trajectory', 'portfolio',
                      'quality', 'risk', 'regulatory').

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
