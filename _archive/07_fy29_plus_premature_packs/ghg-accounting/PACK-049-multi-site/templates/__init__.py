# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Report Templates
=============================================================

This package provides 10 report templates for the PACK-049 GHG Multi-Site
Management Pack. Each template supports multiple rendering formats (Markdown,
HTML, JSON, CSV) with SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. SitePortfolioDashboard      - All-sites overview with KPIs, geography, top emitters
    2. SiteDetailReport            - Individual site drill-down with source breakdown
    3. ConsolidationReport         - Corporate-level consolidated emissions
    4. BoundaryDefinitionReport    - Organisational boundary documentation
    5. RegionalFactorReport        - Factor assignment matrix with tiers and sources
    6. AllocationReport            - Shared services allocation with completeness
    7. SiteComparisonReport        - Cross-site benchmarking league tables
    8. DataCollectionStatusReport  - Submission tracker with completeness matrix
    9. DataQualityReport           - Quality heatmap across 6 dimensions
    10. MultiSiteTrendReport       - Year-over-year corporate and site trends

Author: GreenLang Team
Version: 49.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__pack__ = "PACK-049"
__pack_name__ = "GHG Multi-Site Management Pack"
__templates_count__ = 10

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .site_portfolio_dashboard import SitePortfolioDashboard
except ImportError as e:
    logger.warning("Failed to import SitePortfolioDashboard: %s", e)
    SitePortfolioDashboard = None  # type: ignore[assignment,misc]

try:
    from .site_detail_report import SiteDetailReport
except ImportError as e:
    logger.warning("Failed to import SiteDetailReport: %s", e)
    SiteDetailReport = None  # type: ignore[assignment,misc]

try:
    from .consolidation_report import ConsolidationReport
except ImportError as e:
    logger.warning("Failed to import ConsolidationReport: %s", e)
    ConsolidationReport = None  # type: ignore[assignment,misc]

try:
    from .boundary_definition_report import BoundaryDefinitionReport
except ImportError as e:
    logger.warning("Failed to import BoundaryDefinitionReport: %s", e)
    BoundaryDefinitionReport = None  # type: ignore[assignment,misc]

try:
    from .regional_factor_report import RegionalFactorReport
except ImportError as e:
    logger.warning("Failed to import RegionalFactorReport: %s", e)
    RegionalFactorReport = None  # type: ignore[assignment,misc]

try:
    from .allocation_report import AllocationReport
except ImportError as e:
    logger.warning("Failed to import AllocationReport: %s", e)
    AllocationReport = None  # type: ignore[assignment,misc]

try:
    from .site_comparison_report import SiteComparisonReport
except ImportError as e:
    logger.warning("Failed to import SiteComparisonReport: %s", e)
    SiteComparisonReport = None  # type: ignore[assignment,misc]

try:
    from .data_collection_status_report import DataCollectionStatusReport
except ImportError as e:
    logger.warning("Failed to import DataCollectionStatusReport: %s", e)
    DataCollectionStatusReport = None  # type: ignore[assignment,misc]

try:
    from .data_quality_report import DataQualityReport
except ImportError as e:
    logger.warning("Failed to import DataQualityReport: %s", e)
    DataQualityReport = None  # type: ignore[assignment,misc]

try:
    from .multi_site_trend_report import MultiSiteTrendReport
except ImportError as e:
    logger.warning("Failed to import MultiSiteTrendReport: %s", e)
    MultiSiteTrendReport = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    "SitePortfolioDashboard",
    "SiteDetailReport",
    "ConsolidationReport",
    "BoundaryDefinitionReport",
    "RegionalFactorReport",
    "AllocationReport",
    "SiteComparisonReport",
    "DataCollectionStatusReport",
    "DataQualityReport",
    "MultiSiteTrendReport",
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "site_portfolio_dashboard",
        "class": SitePortfolioDashboard,
        "description": (
            "All-sites portfolio dashboard with summary KPIs, geographic "
            "distribution, facility type breakdown, scope split, top N "
            "emitters, and status overview."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "site_detail_report",
        "class": SiteDetailReport,
        "description": (
            "Individual site drill-down with emissions source breakdown, "
            "intensity KPIs, year-over-year trend, data quality scores, "
            "and emission factor assignments."
        ),
        "category": "site",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "consolidation_report",
        "class": ConsolidationReport,
        "description": (
            "Corporate-level consolidated emissions with scope breakdown, "
            "entity breakdown, intra-group eliminations, equity adjustments, "
            "and bottom-up vs top-down reconciliation."
        ),
        "category": "consolidation",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "boundary_definition_report",
        "class": BoundaryDefinitionReport,
        "description": (
            "Organisational boundary documentation with approach rationale, "
            "entity hierarchy, inclusions/exclusions, materiality assessment, "
            "and year-over-year boundary changes."
        ),
        "category": "boundary",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "regional_factor_report",
        "class": RegionalFactorReport,
        "description": (
            "Emission factor assignment matrix with tier distribution, source "
            "database distribution, grid region mappings, and override tracking."
        ),
        "category": "factors",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "allocation_report",
        "class": AllocationReport,
        "description": (
            "Shared services allocation report with per-service allocation "
            "breakdown, landlord-tenant items, cogeneration items, and "
            "completeness verification."
        ),
        "category": "allocation",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "site_comparison_report",
        "class": SiteComparisonReport,
        "description": (
            "Cross-site benchmarking with league tables, group statistics, "
            "quartile distribution, gap-to-best-practice analysis, and "
            "best practice identification."
        ),
        "category": "comparison",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "data_collection_status_report",
        "class": DataCollectionStatusReport,
        "description": (
            "Data collection submission tracker with per-site status matrix, "
            "completeness by scope, overdue items, data gaps, and estimation "
            "coverage."
        ),
        "category": "collection",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "data_quality_report",
        "class": DataQualityReport,
        "description": (
            "Data quality heatmap across 6 dimensions (completeness, accuracy, "
            "consistency, transparency, timeliness, relevance) with corporate "
            "score, tier distribution, and prioritised remediation actions."
        ),
        "category": "quality",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
    {
        "name": "multi_site_trend_report",
        "class": MultiSiteTrendReport,
        "description": (
            "Multi-year trend analysis with corporate emission trajectory, "
            "scope trends, top site movements, improvement leaders, CAGR, "
            "and structural change tracking."
        ),
        "category": "trend",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "49.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-049 GHG Multi-Site Management report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 report templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_template_names()
        >>> template = registry.get("site_portfolio_dashboard")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
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
        """List all available templates with metadata."""
        return [
            {
                "name": d["name"], "description": d["description"],
                "category": d["category"], "formats": d["formats"],
                "version": d["version"],
            }
            for d in TEMPLATE_CATALOG if d["class"] is not None
        ]

    def list_template_names(self) -> List[str]:
        """List all available template names."""
        return [d["name"] for d in TEMPLATE_CATALOG if d["class"] is not None]

    def get(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get a template instance by name."""
        if name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise KeyError(f"Template '{name}' not found. Available: {available}")

        if config is not None or name not in self._instances:
            template_class = self._templates[name]["class"]
            instance = template_class(config=config)
            if config is None:
                self._instances[name] = instance
            return instance

        return self._instances[name]

    def get_template(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Alias for get()."""
        return self.get(name, config)

    def render(
        self, template_name: str, data: Dict[str, Any],
        format: str = "markdown", config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Render a template in the specified format."""
        template = self.get(template_name, config)
        if format == "markdown":
            return template.render_markdown(data)
        elif format == "html":
            return template.render_html(data)
        elif format == "json":
            return template.render_json(data)
        raise ValueError(f"Unsupported format '{format}'")

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a specific template."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        defn = self._templates[name]
        return {
            "name": defn["name"], "description": defn["description"],
            "category": defn["category"], "formats": defn["formats"],
            "version": defn["version"], "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get templates filtered by category."""
        return [
            {"name": d["name"], "description": d["description"],
             "category": d["category"], "formats": d["formats"], "version": d["version"]}
            for d in TEMPLATE_CATALOG
            if d["category"] == category and d["class"] is not None
        ]

    def has_template(self, name: str) -> bool:
        """Check if a template exists."""
        return name in self._templates

    @property
    def template_count(self) -> int:
        """Return the number of registered templates."""
        return len(self._templates)

    def __repr__(self) -> str:
        return f"TemplateRegistry(templates={self.template_count}, names={self.list_template_names()})"
