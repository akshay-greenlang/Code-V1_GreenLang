# -*- coding: utf-8 -*-
"""
PACK-045 Base Year Management Pack - Report Templates
=========================================================

This package provides 10 report templates for the PACK-045 Base Year
Management Pack. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256 provenance
hashing for audit trail integrity.

Templates:
    1. BaseYearSelectionReport          - Selection criteria, candidate comparison
    2. InventorySummaryReport           - Base year inventory by scope/category/gas
    3. RecalculationTriggerReport       - Detected triggers, significance results
    4. AdjustmentDetailReport           - Adjustment lines, before/after comparison
    5. TimeSeriesDashboard              - Multi-year trends, normalized series
    6. TargetProgressReport             - Target vs actual, SBTi pathway
    7. AuditTrailReport                 - Chronological entries, verification status
    8. PolicyComplianceReport           - Policy settings, framework compliance
    9. MergerAcquisitionReport          - M&A details, pro-rata adjustments
    10. ExecutiveSummaryReport          - 2-page executive summary

Usage:
    >>> from packs.ghg_accounting.PACK_045_base_year.templates import (
    ...     TemplateRegistry,
    ...     BaseYearSelectionReport,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("base_year_selection_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

Author: GreenLang Team
Version: 45.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .base_year_selection_report import BaseYearSelectionReport
except ImportError as e:
    logger.warning("Failed to import BaseYearSelectionReport: %s", e)
    BaseYearSelectionReport = None  # type: ignore[assignment,misc]

try:
    from .inventory_summary_report import InventorySummaryReport
except ImportError as e:
    logger.warning("Failed to import InventorySummaryReport: %s", e)
    InventorySummaryReport = None  # type: ignore[assignment,misc]

try:
    from .recalculation_trigger_report import RecalculationTriggerReport
except ImportError as e:
    logger.warning("Failed to import RecalculationTriggerReport: %s", e)
    RecalculationTriggerReport = None  # type: ignore[assignment,misc]

try:
    from .adjustment_detail_report import AdjustmentDetailReport
except ImportError as e:
    logger.warning("Failed to import AdjustmentDetailReport: %s", e)
    AdjustmentDetailReport = None  # type: ignore[assignment,misc]

try:
    from .time_series_dashboard import TimeSeriesDashboard
except ImportError as e:
    logger.warning("Failed to import TimeSeriesDashboard: %s", e)
    TimeSeriesDashboard = None  # type: ignore[assignment,misc]

try:
    from .target_progress_report import TargetProgressReport
except ImportError as e:
    logger.warning("Failed to import TargetProgressReport: %s", e)
    TargetProgressReport = None  # type: ignore[assignment,misc]

try:
    from .audit_trail_report import AuditTrailReport
except ImportError as e:
    logger.warning("Failed to import AuditTrailReport: %s", e)
    AuditTrailReport = None  # type: ignore[assignment,misc]

try:
    from .policy_compliance_report import PolicyComplianceReport
except ImportError as e:
    logger.warning("Failed to import PolicyComplianceReport: %s", e)
    PolicyComplianceReport = None  # type: ignore[assignment,misc]

try:
    from .merger_acquisition_report import MergerAcquisitionReport
except ImportError as e:
    logger.warning("Failed to import MergerAcquisitionReport: %s", e)
    MergerAcquisitionReport = None  # type: ignore[assignment,misc]

try:
    from .executive_summary_report import ExecutiveSummaryReport
except ImportError as e:
    logger.warning("Failed to import ExecutiveSummaryReport: %s", e)
    ExecutiveSummaryReport = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "BaseYearSelectionReport",
    "InventorySummaryReport",
    "RecalculationTriggerReport",
    "AdjustmentDetailReport",
    "TimeSeriesDashboard",
    "TargetProgressReport",
    "AuditTrailReport",
    "PolicyComplianceReport",
    "MergerAcquisitionReport",
    "ExecutiveSummaryReport",
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
    Type[BaseYearSelectionReport],
    Type[InventorySummaryReport],
    Type[RecalculationTriggerReport],
    Type[AdjustmentDetailReport],
    Type[TimeSeriesDashboard],
    Type[TargetProgressReport],
    Type[AuditTrailReport],
    Type[PolicyComplianceReport],
    Type[MergerAcquisitionReport],
    Type[ExecutiveSummaryReport],
]

TemplateInstance = Union[
    BaseYearSelectionReport,
    InventorySummaryReport,
    RecalculationTriggerReport,
    AdjustmentDetailReport,
    TimeSeriesDashboard,
    TargetProgressReport,
    AuditTrailReport,
    PolicyComplianceReport,
    MergerAcquisitionReport,
    ExecutiveSummaryReport,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "base_year_selection_report",
        "class": BaseYearSelectionReport,
        "description": (
            "Base year selection report with candidate year evaluation, "
            "weighted scoring matrices, recommendation rationale, and "
            "stakeholder sign-off tracking."
        ),
        "category": "selection",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "inventory_summary_report",
        "class": InventorySummaryReport,
        "description": (
            "Base year inventory summary showing emissions by scope, "
            "category, and gas type with totals, percentage breakdowns, "
            "methodology references, and data quality indicators."
        ),
        "category": "inventory",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "recalculation_trigger_report",
        "class": RecalculationTriggerReport,
        "description": (
            "Recalculation trigger report with detected triggers, "
            "significance assessment results, threshold analysis, and "
            "prioritized action recommendations."
        ),
        "category": "triggers",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "adjustment_detail_report",
        "class": AdjustmentDetailReport,
        "description": (
            "Adjustment detail report with individual line items, "
            "before/after emission comparisons by scope, approval "
            "workflow status, and net impact analysis."
        ),
        "category": "adjustments",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "time_series_dashboard",
        "class": TimeSeriesDashboard,
        "description": (
            "Time series dashboard with multi-year emission trends, "
            "scope-level annual breakdowns, normalized intensity series, "
            "consistency findings, and year-over-year variance analysis."
        ),
        "category": "trends",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "target_progress_report",
        "class": TargetProgressReport,
        "description": (
            "Target progress report with target vs actual comparison, "
            "SBTi pathway alignment tracking, reduction attribution "
            "by initiative, and gap-to-target projections."
        ),
        "category": "targets",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "audit_trail_report",
        "class": AuditTrailReport,
        "description": (
            "Audit trail report with chronological entries, approval "
            "records, verification status tracking, and SHA-256 "
            "integrity chain for data provenance."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "policy_compliance_report",
        "class": PolicyComplianceReport,
        "description": (
            "Policy compliance report with base year policy settings, "
            "multi-framework compliance matrices (GHG Protocol, ISO 14064, "
            "SBTi, CSRD), gap analysis, and remediation recommendations."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "merger_acquisition_report",
        "class": MergerAcquisitionReport,
        "description": (
            "Merger and acquisition impact report with entity details, "
            "emission impact analysis, pro-rata adjustment calculations, "
            "and base year recalculation recommendations."
        ),
        "category": "structural",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReport,
        "description": (
            "Two-page executive summary with base year status overview, "
            "key emission metrics, recent changes, target progress "
            "highlights, and strategic recommendations."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "45.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-045 Base Year Management report templates.

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
        >>> template = registry.get("base_year_selection_report")
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
            name: Template name (e.g., 'base_year_selection_report').
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
            category: Category string (e.g., 'selection', 'inventory',
                      'triggers', 'adjustments', 'trends', 'targets',
                      'audit', 'compliance', 'structural', 'executive').

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
