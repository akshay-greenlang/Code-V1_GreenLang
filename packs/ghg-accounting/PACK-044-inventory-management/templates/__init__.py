# -*- coding: utf-8 -*-
"""
PACK-044 GHG Inventory Management Pack - Report Templates
============================================================

This package provides 10 report templates for the PACK-044 GHG Inventory
Management Pack. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256 provenance
hashing for audit trail integrity.

Templates:
    1. InventoryStatusDashboard          - Period status KPIs, progress bars
    2. DataCollectionTracker             - Submission rates, overdue, coverage
    3. QualityScorecard                  - QA/QC pass/fail, quality scores
    4. ChangeLogReport                   - Changes, impacts, approvals
    5. ReviewSummaryReport               - Review decisions, comments
    6. VersionComparisonReport           - Version diffs
    7. ConsolidationStatusReport         - Entity submissions, totals
    8. GapAnalysisReport                 - Gaps, recommendations
    9. DocumentationIndex                - Doc completeness
    10. BenchmarkingReport               - Peer comparison, rankings

Usage:
    >>> from packs.ghg_accounting.PACK_044_inventory_management.templates import (
    ...     TemplateRegistry,
    ...     InventoryStatusDashboard,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("inventory_status_dashboard")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

Author: GreenLang Team
Version: 44.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .inventory_status_dashboard import InventoryStatusDashboard
except ImportError as e:
    logger.warning("Failed to import InventoryStatusDashboard: %s", e)
    InventoryStatusDashboard = None  # type: ignore[assignment,misc]

try:
    from .data_collection_tracker import DataCollectionTracker
except ImportError as e:
    logger.warning("Failed to import DataCollectionTracker: %s", e)
    DataCollectionTracker = None  # type: ignore[assignment,misc]

try:
    from .quality_scorecard import QualityScorecard
except ImportError as e:
    logger.warning("Failed to import QualityScorecard: %s", e)
    QualityScorecard = None  # type: ignore[assignment,misc]

try:
    from .change_log_report import ChangeLogReport
except ImportError as e:
    logger.warning("Failed to import ChangeLogReport: %s", e)
    ChangeLogReport = None  # type: ignore[assignment,misc]

try:
    from .review_summary_report import ReviewSummaryReport
except ImportError as e:
    logger.warning("Failed to import ReviewSummaryReport: %s", e)
    ReviewSummaryReport = None  # type: ignore[assignment,misc]

try:
    from .version_comparison_report import VersionComparisonReport
except ImportError as e:
    logger.warning("Failed to import VersionComparisonReport: %s", e)
    VersionComparisonReport = None  # type: ignore[assignment,misc]

try:
    from .consolidation_status_report import ConsolidationStatusReport
except ImportError as e:
    logger.warning("Failed to import ConsolidationStatusReport: %s", e)
    ConsolidationStatusReport = None  # type: ignore[assignment,misc]

try:
    from .gap_analysis_report import GapAnalysisReport
except ImportError as e:
    logger.warning("Failed to import GapAnalysisReport: %s", e)
    GapAnalysisReport = None  # type: ignore[assignment,misc]

try:
    from .documentation_index import DocumentationIndex
except ImportError as e:
    logger.warning("Failed to import DocumentationIndex: %s", e)
    DocumentationIndex = None  # type: ignore[assignment,misc]

try:
    from .benchmarking_report import BenchmarkingReport
except ImportError as e:
    logger.warning("Failed to import BenchmarkingReport: %s", e)
    BenchmarkingReport = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "InventoryStatusDashboard",
    "DataCollectionTracker",
    "QualityScorecard",
    "ChangeLogReport",
    "ReviewSummaryReport",
    "VersionComparisonReport",
    "ConsolidationStatusReport",
    "GapAnalysisReport",
    "DocumentationIndex",
    "BenchmarkingReport",
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
    Type[InventoryStatusDashboard],
    Type[DataCollectionTracker],
    Type[QualityScorecard],
    Type[ChangeLogReport],
    Type[ReviewSummaryReport],
    Type[VersionComparisonReport],
    Type[ConsolidationStatusReport],
    Type[GapAnalysisReport],
    Type[DocumentationIndex],
    Type[BenchmarkingReport],
]

TemplateInstance = Union[
    InventoryStatusDashboard,
    DataCollectionTracker,
    QualityScorecard,
    ChangeLogReport,
    ReviewSummaryReport,
    VersionComparisonReport,
    ConsolidationStatusReport,
    GapAnalysisReport,
    DocumentationIndex,
    BenchmarkingReport,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "inventory_status_dashboard",
        "class": InventoryStatusDashboard,
        "description": (
            "Period status dashboard with KPIs, progress bars for each "
            "inventory phase, milestone tracking, scope readiness summaries, "
            "and prioritized action items."
        ),
        "category": "management",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "data_collection_tracker",
        "class": DataCollectionTracker,
        "description": (
            "Data collection tracking with submission rates per facility, "
            "overdue data submissions list, coverage analysis by scope and "
            "category, and data freshness metrics."
        ),
        "category": "collection",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "quality_scorecard",
        "class": QualityScorecard,
        "description": (
            "QA/QC quality scorecard with pass/fail rates, quality scores "
            "by scope and category, DQI breakdowns with weights, and "
            "prioritized improvement recommendations."
        ),
        "category": "quality",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "change_log_report",
        "class": ChangeLogReport,
        "description": (
            "Change log report with inventory changes, emission impact "
            "analysis (before/after tCO2e), approval workflows, and "
            "categorized change tracking."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "review_summary_report",
        "class": ReviewSummaryReport,
        "description": (
            "Review summary with review cycle metadata, reviewer decisions "
            "per scope, review comments and issues with severity, and "
            "sign-off status tracking."
        ),
        "category": "review",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "version_comparison_report",
        "class": VersionComparisonReport,
        "description": (
            "Version comparison between two inventory versions with emission "
            "differences by scope/category, methodology changes, data source "
            "changes, and emission factor changes."
        ),
        "category": "versioning",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "consolidation_status_report",
        "class": ConsolidationStatusReport,
        "description": (
            "Consolidation status with entity submission progress, entity-level "
            "emission totals, inter-company eliminations, and consolidated "
            "group-level totals."
        ),
        "category": "consolidation",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "gap_analysis_report",
        "class": GapAnalysisReport,
        "description": (
            "Gap analysis identifying data gaps, methodology gaps, coverage "
            "gaps with severity assessments, estimated emission impact, and "
            "prioritized recommendations with effort/timeline."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "documentation_index",
        "class": DocumentationIndex,
        "description": (
            "Documentation completeness tracker with document inventory, "
            "completeness by category, missing document identification, "
            "and compliance documentation requirements mapping."
        ),
        "category": "documentation",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
    {
        "name": "benchmarking_report",
        "class": BenchmarkingReport,
        "description": (
            "Peer benchmarking with intensity metrics comparison, sector "
            "rankings, best practices from top performers, and improvement "
            "opportunity identification with gap analysis."
        ),
        "category": "benchmarking",
        "formats": ["markdown", "html", "json"],
        "version": "44.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-044 GHG Inventory Management report templates.

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
        >>> template = registry.get("inventory_status_dashboard")
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
            name: Template name (e.g., 'inventory_status_dashboard').
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
            category: Category string (e.g., 'management', 'collection',
                      'quality', 'audit', 'review', 'versioning',
                      'consolidation', 'analysis', 'documentation',
                      'benchmarking').

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
