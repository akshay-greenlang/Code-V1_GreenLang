# -*- coding: utf-8 -*-
"""
PACK-036 Utility Analysis Pack - Report Templates
=========================================================

This package provides 10 utility analysis report templates for the
PACK-036 Utility Analysis Pack. Each template supports three
rendering formats: Markdown, HTML (with inline CSS), and JSON. All
templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. BillAuditReportTemplate              - Bill audit findings report
    2. RateComparisonReportTemplate          - Rate structure comparison report
    3. DemandProfileReportTemplate           - Load profile and demand analysis
    4. CostAllocationReportTemplate          - Cost allocation by entity report
    5. BudgetForecastReportTemplate          - Multi-scenario budget forecast
    6. ProcurementStrategyReportTemplate     - Procurement strategy report
    7. BenchmarkReportTemplate              - Peer benchmarking report
    8. RegulatoryChargeReportTemplate       - Regulatory surcharge analysis
    9. ExecutiveDashboardTemplate           - C-suite utility KPI dashboard
    10. UtilitySavingsReportTemplate         - Savings tracking and verification

Usage:
    >>> from packs.energy_efficiency.PACK_036_utility_analysis.templates import (
    ...     TemplateRegistry,
    ...     BillAuditReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("bill_audit_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 36.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .bill_audit_report import BillAuditReportTemplate
from .rate_comparison_report import RateComparisonReportTemplate
from .demand_profile_report import DemandProfileReportTemplate
from .cost_allocation_report import CostAllocationReportTemplate
from .budget_forecast_report import BudgetForecastReportTemplate
from .procurement_strategy_report import ProcurementStrategyReportTemplate
from .benchmark_report import BenchmarkReportTemplate
from .regulatory_charge_report import RegulatoryChargeReportTemplate
from .executive_dashboard import ExecutiveDashboardTemplate
from .utility_savings_report import UtilitySavingsReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "BillAuditReportTemplate",
    "RateComparisonReportTemplate",
    "DemandProfileReportTemplate",
    "CostAllocationReportTemplate",
    "BudgetForecastReportTemplate",
    "ProcurementStrategyReportTemplate",
    "BenchmarkReportTemplate",
    "RegulatoryChargeReportTemplate",
    "ExecutiveDashboardTemplate",
    "UtilitySavingsReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[BillAuditReportTemplate],
    Type[RateComparisonReportTemplate],
    Type[DemandProfileReportTemplate],
    Type[CostAllocationReportTemplate],
    Type[BudgetForecastReportTemplate],
    Type[ProcurementStrategyReportTemplate],
    Type[BenchmarkReportTemplate],
    Type[RegulatoryChargeReportTemplate],
    Type[ExecutiveDashboardTemplate],
    Type[UtilitySavingsReportTemplate],
]

TemplateInstance = Union[
    BillAuditReportTemplate,
    RateComparisonReportTemplate,
    DemandProfileReportTemplate,
    CostAllocationReportTemplate,
    BudgetForecastReportTemplate,
    ProcurementStrategyReportTemplate,
    BenchmarkReportTemplate,
    RegulatoryChargeReportTemplate,
    ExecutiveDashboardTemplate,
    UtilitySavingsReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "bill_audit_report",
        "class": BillAuditReportTemplate,
        "description": (
            "Utility bill audit findings report with executive summary, "
            "bill-by-bill analysis, line-item validation results, detected "
            "errors and overcharges, historical billing trends, estimated "
            "refund amounts, and corrective action recommendations."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "rate_comparison_report",
        "class": RateComparisonReportTemplate,
        "description": (
            "Rate structure comparison report with current tariff analysis, "
            "alternative rate simulation results, TOU optimization potential, "
            "demand charge impact comparison, annual cost projections under "
            "each rate, and ranked recommendations with transition guidance."
        ),
        "category": "rates",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "demand_profile_report",
        "class": DemandProfileReportTemplate,
        "description": (
            "Load profile and demand analysis report with peak demand "
            "identification, load factor analysis, demand ratchet impact, "
            "coincident vs non-coincident peak comparison, load duration "
            "curves, demand response opportunity assessment, and power "
            "factor correction recommendations."
        ),
        "category": "demand",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "cost_allocation_report",
        "class": CostAllocationReportTemplate,
        "description": (
            "Cost allocation report with meter-to-entity mapping, "
            "consumption disaggregation by department/tenant/process, "
            "demand charge allocation methodology, shared services "
            "apportionment, chargeback summaries, and variance analysis "
            "against budgeted allocations."
        ),
        "category": "allocation",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "budget_forecast_report",
        "class": BudgetForecastReportTemplate,
        "description": (
            "Multi-scenario budget forecast report with historical trend "
            "analysis, weather-normalized baseline, rate escalation "
            "projections, scenario comparison (base/optimistic/pessimistic), "
            "monthly and annual forecasts, variance sensitivity analysis, "
            "and budget recommendation with confidence intervals."
        ),
        "category": "budget",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "procurement_strategy_report",
        "class": ProcurementStrategyReportTemplate,
        "description": (
            "Procurement strategy report with current contract review, "
            "market price benchmarking, supplier comparison matrix, "
            "contract term optimization, renewable procurement options, "
            "hedging strategy assessment, and RFP framework with "
            "evaluation criteria."
        ),
        "category": "procurement",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "benchmark_report",
        "class": BenchmarkReportTemplate,
        "description": (
            "Peer benchmarking report with EUI comparison against peer "
            "buildings, cost-per-square-meter analysis, Energy Star score "
            "equivalent, CIBSE TM46 benchmark alignment, percentile "
            "ranking, performance gap analysis, and improvement target "
            "recommendations."
        ),
        "category": "benchmark",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "regulatory_charge_report",
        "class": RegulatoryChargeReportTemplate,
        "description": (
            "Regulatory surcharge analysis report with itemized regulatory "
            "charge breakdown, network charge analysis, capacity charge "
            "optimization, tax and levy assessment, exemption eligibility "
            "review, and cost reduction opportunities through regulatory "
            "charge management."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "executive_dashboard",
        "class": ExecutiveDashboardTemplate,
        "description": (
            "C-suite executive utility KPI dashboard with total utility "
            "spend, cost per unit area, year-over-year trends, budget "
            "variance, savings achieved, rate optimization status, "
            "procurement timeline, benchmark position, and top action "
            "items for board-level presentation."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
    {
        "name": "utility_savings_report",
        "class": UtilitySavingsReportTemplate,
        "description": (
            "Utility savings tracking and verification report with "
            "implemented savings summary, IPMVP-aligned verification, "
            "avoided cost calculations, weather-normalized comparisons, "
            "cumulative savings trends, measure-level performance, and "
            "ROI achievement against targets."
        ),
        "category": "savings",
        "formats": ["markdown", "html", "json"],
        "version": "36.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-036 Utility Analysis report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 utility analysis report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("bill_audit_report")
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
            name: Template name (e.g., 'bill_audit_report').
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
            category: Category string (e.g., 'audit', 'rates',
                      'demand', 'allocation', 'budget', 'procurement',
                      'benchmark', 'regulatory', 'executive', 'savings').

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
