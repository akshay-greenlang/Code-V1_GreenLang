# -*- coding: utf-8 -*-
"""
PACK-038 Peak Shaving Pack - Report Templates
=====================================================

This package provides 10 peak shaving report templates for the
PACK-038 Peak Shaving Pack. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates
include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. LoadProfileReportTemplate            - Load profile analysis
    2. PeakAnalysisReportTemplate           - Peak identification and analysis
    3. DemandChargeReportTemplate           - Demand charge decomposition
    4. BESSSizingReportTemplate             - BESS optimization and sizing
    5. LoadShiftingReportTemplate           - Load shifting plan
    6. CPManagementReportTemplate           - Coincident peak management
    7. FinancialAnalysisReportTemplate      - Financial analysis (NPV/IRR/MC)
    8. PowerFactorReportTemplate            - Power factor analysis
    9. ExecutiveSummaryReportTemplate       - C-suite summary
    10. VerificationReportTemplate          - M&V verification

Usage:
    >>> from packs.energy_efficiency.PACK_038_peak_shaving.templates import (
    ...     TemplateRegistry,
    ...     LoadProfileReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("load_profile_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 38.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .load_profile_report import LoadProfileReportTemplate
from .peak_analysis_report import PeakAnalysisReportTemplate
from .demand_charge_report import DemandChargeReportTemplate
from .bess_sizing_report import BESSSizingReportTemplate
from .load_shifting_report import LoadShiftingReportTemplate
from .cp_management_report import CPManagementReportTemplate
from .financial_analysis_report import FinancialAnalysisReportTemplate
from .power_factor_report import PowerFactorReportTemplate
from .executive_summary_report import ExecutiveSummaryReportTemplate
from .verification_report import VerificationReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "LoadProfileReportTemplate",
    "PeakAnalysisReportTemplate",
    "DemandChargeReportTemplate",
    "BESSSizingReportTemplate",
    "LoadShiftingReportTemplate",
    "CPManagementReportTemplate",
    "FinancialAnalysisReportTemplate",
    "PowerFactorReportTemplate",
    "ExecutiveSummaryReportTemplate",
    "VerificationReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[LoadProfileReportTemplate],
    Type[PeakAnalysisReportTemplate],
    Type[DemandChargeReportTemplate],
    Type[BESSSizingReportTemplate],
    Type[LoadShiftingReportTemplate],
    Type[CPManagementReportTemplate],
    Type[FinancialAnalysisReportTemplate],
    Type[PowerFactorReportTemplate],
    Type[ExecutiveSummaryReportTemplate],
    Type[VerificationReportTemplate],
]

TemplateInstance = Union[
    LoadProfileReportTemplate,
    PeakAnalysisReportTemplate,
    DemandChargeReportTemplate,
    BESSSizingReportTemplate,
    LoadShiftingReportTemplate,
    CPManagementReportTemplate,
    FinancialAnalysisReportTemplate,
    PowerFactorReportTemplate,
    ExecutiveSummaryReportTemplate,
    VerificationReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "load_profile_report",
        "class": LoadProfileReportTemplate,
        "description": (
            "Load profile analysis report showing demand duration curves, "
            "load factor analysis across on-peak/off-peak/shoulder periods, "
            "day-type consumption patterns (weekday/weekend/holiday), "
            "seasonal profile decomposition, and anomaly detection summary."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "peak_analysis_report",
        "class": PeakAnalysisReportTemplate,
        "description": (
            "Peak identification report with top-N demand peak ranking, "
            "attribution breakdown by load category and coincidence factor, "
            "clustering analysis of coincident peaks, avoidability assessment "
            "with confidence scoring, and shaving potential estimates."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "demand_charge_report",
        "class": DemandChargeReportTemplate,
        "description": (
            "Demand charge decomposition report showing facility, transmission, "
            "and distribution component breakdown, marginal demand values at "
            "various reduction levels, tariff comparison across rate schedules, "
            "projected charges under peak reduction scenarios, and rate optimization."
        ),
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "bess_sizing_report",
        "class": BESSSizingReportTemplate,
        "description": (
            "BESS optimization report with technology comparison across "
            "lithium-ion, flow, and sodium-ion chemistries, dispatch simulation "
            "results for peak shaving scenarios, degradation projections over "
            "system lifetime, and detailed financial analysis with NPV/IRR."
        ),
        "category": "engineering",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "load_shifting_report",
        "class": LoadShiftingReportTemplate,
        "description": (
            "Load shifting plan report showing shiftable load inventory with "
            "flexibility windows, constraint summaries including process "
            "dependencies and safety limits, optimized dispatch schedule, "
            "and rebound effect analysis with mitigation strategies."
        ),
        "category": "operations",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "cp_management_report",
        "class": CPManagementReportTemplate,
        "description": (
            "Coincident peak management report showing CP prediction accuracy "
            "metrics, demand response performance during CP events, charge "
            "allocation impact analysis with capacity tag savings, and annual "
            "CP day forecasts with confidence intervals."
        ),
        "category": "operations",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "financial_analysis_report",
        "class": FinancialAnalysisReportTemplate,
        "description": (
            "Financial analysis report with NPV, IRR, and payback calculations, "
            "incentive and rebate capture analysis, revenue stacking across "
            "demand charges, arbitrage, and ancillary services, sensitivity "
            "analysis, and Monte Carlo simulation probability distributions."
        ),
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "power_factor_report",
        "class": PowerFactorReportTemplate,
        "description": (
            "Power factor analysis report showing PF profile across billing "
            "periods, reactive demand quantification by source, correction "
            "equipment sizing recommendations, penalty savings calculations, "
            "and harmonic distortion assessment with THD metrics."
        ),
        "category": "engineering",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "2-4 page C-suite executive summary with key peak shaving metrics "
            "dashboard, demand charge savings achieved and projected, BESS ROI "
            "summary, recommended actions with priority and timeline, and "
            "year-over-year performance comparison for board presentation."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
    {
        "name": "verification_report",
        "class": VerificationReportTemplate,
        "description": (
            "M&V verification report with baseline comparison using adjusted "
            "baselines, savings verification through IPMVP-compliant methods, "
            "regulatory compliance documentation, performance event log, "
            "statistical analysis with confidence intervals, and certifications."
        ),
        "category": "verification",
        "formats": ["markdown", "html", "json"],
        "version": "38.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-038 Peak Shaving report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 peak shaving report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("load_profile_report")
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
            name: Template name (e.g., 'load_profile_report').
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
                      'engineering', 'operations', 'executive',
                      'verification').

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
