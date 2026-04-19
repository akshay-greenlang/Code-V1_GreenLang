# -*- coding: utf-8 -*-
"""
PACK-039 Energy Monitoring Pack - Report Templates
=====================================================

This package provides 10 energy monitoring report templates for the
PACK-039 Energy Monitoring Pack. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates
include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. MeterInventoryReportTemplate          - Meter registry and hierarchy
    2. EnergyConsumptionReportTemplate       - Consumption dashboard
    3. AnomalyReportTemplate                 - Detected anomalies
    4. EnPIPerformanceReportTemplate         - EnPI tracking
    5. CostAllocationReportTemplate          - Cost allocation by tenant/dept
    6. BudgetVarianceReportTemplate          - Budget vs actual
    7. AlarmSummaryReportTemplate            - Alarm management metrics
    8. UtilityBillReportTemplate             - Utility bill validation
    9. ExecutiveSummaryReportTemplate        - C-suite summary
    10. ISO50001ComplianceReportTemplate     - ISO 50001 management review

Usage:
    >>> from packs.energy_efficiency.PACK_039_energy_monitoring.templates import (
    ...     TemplateRegistry,
    ...     MeterInventoryReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("meter_inventory_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 39.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .meter_inventory_report import MeterInventoryReportTemplate
from .energy_consumption_report import EnergyConsumptionReportTemplate
from .anomaly_report import AnomalyReportTemplate
from .enpi_performance_report import EnPIPerformanceReportTemplate
from .cost_allocation_report import CostAllocationReportTemplate
from .budget_variance_report import BudgetVarianceReportTemplate
from .alarm_summary_report import AlarmSummaryReportTemplate
from .utility_bill_report import UtilityBillReportTemplate
from .executive_summary_report import ExecutiveSummaryReportTemplate
from .iso50001_compliance_report import ISO50001ComplianceReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "MeterInventoryReportTemplate",
    "EnergyConsumptionReportTemplate",
    "AnomalyReportTemplate",
    "EnPIPerformanceReportTemplate",
    "CostAllocationReportTemplate",
    "BudgetVarianceReportTemplate",
    "AlarmSummaryReportTemplate",
    "UtilityBillReportTemplate",
    "ExecutiveSummaryReportTemplate",
    "ISO50001ComplianceReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[MeterInventoryReportTemplate],
    Type[EnergyConsumptionReportTemplate],
    Type[AnomalyReportTemplate],
    Type[EnPIPerformanceReportTemplate],
    Type[CostAllocationReportTemplate],
    Type[BudgetVarianceReportTemplate],
    Type[AlarmSummaryReportTemplate],
    Type[UtilityBillReportTemplate],
    Type[ExecutiveSummaryReportTemplate],
    Type[ISO50001ComplianceReportTemplate],
]

TemplateInstance = Union[
    MeterInventoryReportTemplate,
    EnergyConsumptionReportTemplate,
    AnomalyReportTemplate,
    EnPIPerformanceReportTemplate,
    CostAllocationReportTemplate,
    BudgetVarianceReportTemplate,
    AlarmSummaryReportTemplate,
    UtilityBillReportTemplate,
    ExecutiveSummaryReportTemplate,
    ISO50001ComplianceReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "meter_inventory_report",
        "class": MeterInventoryReportTemplate,
        "description": (
            "Meter registry report showing hierarchical meter topology, "
            "calibration status tracking with drift analysis, communication "
            "protocol summary with success rates, zone/system coverage "
            "analysis with gap identification, and per-meter data quality metrics."
        ),
        "category": "inventory",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "energy_consumption_report",
        "class": EnergyConsumptionReportTemplate,
        "description": (
            "Consumption dashboard report showing trend analysis over time, "
            "breakdown by system/building/fuel type with EUI metrics, load "
            "profile analysis with peak/base/shoulder periods, weather-normalized "
            "consumption overlay with HDD/CDD correlation, and efficiency tracking."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "anomaly_report",
        "class": AnomalyReportTemplate,
        "description": (
            "Anomaly detection report showing detected energy anomalies with "
            "severity classification, root cause analysis with confidence "
            "scoring, estimated waste quantification by system, investigation "
            "status tracking, and resolution action management with savings."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "enpi_performance_report",
        "class": EnPIPerformanceReportTemplate,
        "description": (
            "EnPI tracking report showing regression model results with "
            "coefficient analysis, CUSUM cumulative savings charts, "
            "statistical significance testing with confidence intervals, "
            "baseline comparison with adjustment factors, and improvement "
            "percentage tracking across reporting periods."
        ),
        "category": "performance",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "cost_allocation_report",
        "class": CostAllocationReportTemplate,
        "description": (
            "Cost allocation report showing distribution by tenant and "
            "department with metered consumption data, demand contribution "
            "analysis with diversity factors, rate tier analysis, "
            "reconciliation to utility bills, and charge-back summary."
        ),
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "budget_variance_report",
        "class": BudgetVarianceReportTemplate,
        "description": (
            "Budget vs actual report showing variance decomposition into "
            "weather, volume, and efficiency components, cumulative tracking "
            "over fiscal periods, forecast-to-completion projections with "
            "confidence intervals, and department-level variance analysis."
        ),
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "alarm_summary_report",
        "class": AlarmSummaryReportTemplate,
        "description": (
            "Alarm management report showing KPIs including MTTA, MTTR, "
            "and false alarm rate against ISA-18.2 targets, priority "
            "distribution analysis, standing alarm inventory with duration, "
            "top recurring alarm patterns with waste estimates, and operator "
            "performance metrics."
        ),
        "category": "operations",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "utility_bill_report",
        "class": UtilityBillReportTemplate,
        "description": (
            "Utility bill validation report showing meter-to-bill "
            "reconciliation with variance analysis, rate schedule analysis "
            "with tier breakdown, billing error detection with impact "
            "quantification, historical bill comparison, and charge "
            "component decomposition."
        ),
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "2-4 page C-suite executive summary with energy KPI dashboard "
            "showing consumption, cost, EnPI, and carbon metrics, cost "
            "trend analysis, EnPI performance highlights, top anomalies "
            "requiring attention, savings achieved, prioritized action "
            "items, and outlook with forecast."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
    {
        "name": "iso50001_compliance_report",
        "class": ISO50001ComplianceReportTemplate,
        "description": (
            "ISO 50001 management review report showing EnPI results against "
            "objectives and targets, energy baseline status with adjustment "
            "factors, significant energy use analysis, improvement action "
            "register with savings tracking, and conformity assessment with "
            "clause-level findings."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": "39.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-039 Energy Monitoring report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 energy monitoring report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("meter_inventory_report")
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
            name: Template name (e.g., 'meter_inventory_report').
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
            category: Category string (e.g., 'analysis', 'financial',
                      'operations', 'performance', 'inventory',
                      'executive', 'compliance').

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
