# -*- coding: utf-8 -*-
"""
PACK-040 Measurement & Verification (M&V) Pack - Report Templates
====================================================================

This package provides 10 M&V report templates for the PACK-040
Measurement & Verification Pack. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates
include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. MVPlanReportTemplate               - M&V plan documentation
    2. BaselineReportTemplate             - Baseline analysis
    3. SavingsReportTemplate              - Savings verification
    4. UncertaintyReportTemplate          - Uncertainty analysis
    5. AnnualMVReportTemplate             - Annual M&V summary
    6. OptionComparisonReportTemplate     - IPMVP option comparison
    7. MeteringPlanReportTemplate         - Metering plan
    8. PersistenceReportTemplate          - Persistence tracking
    9. ExecutiveSummaryReportTemplate     - Executive summary
    10. ComplianceReportTemplate          - Standards compliance

Usage:
    >>> from packs.energy_efficiency.PACK_040_mv.templates import (
    ...     TemplateRegistry,
    ...     MVPlanReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("mv_plan_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 40.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .mv_plan_report import MVPlanReportTemplate
from .baseline_report import BaselineReportTemplate
from .savings_report import SavingsReportTemplate
from .uncertainty_report import UncertaintyReportTemplate
from .annual_mv_report import AnnualMVReportTemplate
from .option_comparison_report import OptionComparisonReportTemplate
from .metering_plan_report import MeteringPlanReportTemplate
from .persistence_report import PersistenceReportTemplate
from .executive_summary_report import ExecutiveSummaryReportTemplate
from .compliance_report import ComplianceReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "MVPlanReportTemplate",
    "BaselineReportTemplate",
    "SavingsReportTemplate",
    "UncertaintyReportTemplate",
    "AnnualMVReportTemplate",
    "OptionComparisonReportTemplate",
    "MeteringPlanReportTemplate",
    "PersistenceReportTemplate",
    "ExecutiveSummaryReportTemplate",
    "ComplianceReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[MVPlanReportTemplate],
    Type[BaselineReportTemplate],
    Type[SavingsReportTemplate],
    Type[UncertaintyReportTemplate],
    Type[AnnualMVReportTemplate],
    Type[OptionComparisonReportTemplate],
    Type[MeteringPlanReportTemplate],
    Type[PersistenceReportTemplate],
    Type[ExecutiveSummaryReportTemplate],
    Type[ComplianceReportTemplate],
]

TemplateInstance = Union[
    MVPlanReportTemplate,
    BaselineReportTemplate,
    SavingsReportTemplate,
    UncertaintyReportTemplate,
    AnnualMVReportTemplate,
    OptionComparisonReportTemplate,
    MeteringPlanReportTemplate,
    PersistenceReportTemplate,
    ExecutiveSummaryReportTemplate,
    ComplianceReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "mv_plan_report",
        "class": MVPlanReportTemplate,
        "description": (
            "M&V plan report showing ECM description with interaction "
            "effects, IPMVP option selection rationale, measurement "
            "boundary definition with energy streams, baseline and "
            "reporting period specification, metering plan with "
            "calibration schedule, and adjustment methodology."
        ),
        "category": "planning",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "baseline_report",
        "class": BaselineReportTemplate,
        "description": (
            "Baseline analysis report showing regression model results "
            "with coefficient analysis, ASHRAE Guideline 14 statistical "
            "validation (CVRMSE, NMBE, R-squared), independent variable "
            "significance testing, residual diagnostics with normality "
            "and heteroscedasticity checks, change-point model segments, "
            "and model selection rationale with AIC/BIC comparison."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "savings_report",
        "class": SavingsReportTemplate,
        "description": (
            "Savings verification report showing adjusted baseline "
            "versus actual consumption comparison, avoided energy "
            "calculation with routine and non-routine adjustments, "
            "cost savings with ROI and payback metrics, uncertainty "
            "bounds at specified confidence levels, cumulative savings "
            "tracking, and ECM realization rate analysis."
        ),
        "category": "verification",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "uncertainty_report",
        "class": UncertaintyReportTemplate,
        "description": (
            "Uncertainty analysis report showing measurement uncertainty "
            "by meter with bias and random components, model uncertainty "
            "from regression diagnostics, sampling uncertainty with "
            "finite population correction, combined fractional savings "
            "uncertainty (FSU) per ASHRAE 14, minimum detectable savings "
            "calculation, and sensitivity analysis with tornado chart."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "annual_mv_report",
        "class": AnnualMVReportTemplate,
        "description": (
            "Annual M&V summary report showing year-to-date savings "
            "with monthly detail, cumulative multi-year savings, trend "
            "analysis with performance indicators, ECM-level performance "
            "tracking with realization rates, compliance status against "
            "targets, and model performance validation."
        ),
        "category": "reporting",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "option_comparison_report",
        "class": OptionComparisonReportTemplate,
        "description": (
            "IPMVP option comparison report showing detailed assessment "
            "of Options A, B, C, and D with applicability scoring, "
            "weighted suitability criteria matrix, cost-effectiveness "
            "analysis with M&V-to-savings ratio, accuracy-versus-cost "
            "trade-offs, and final recommendation with rationale."
        ),
        "category": "planning",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "metering_plan_report",
        "class": MeteringPlanReportTemplate,
        "description": (
            "Metering plan report showing meter inventory with full "
            "specifications, calibration schedule tracking with drift "
            "analysis, sampling protocol details with confidence and "
            "precision targets, data collection and management procedures, "
            "quality assurance checks, and communication infrastructure."
        ),
        "category": "planning",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "persistence_report",
        "class": PersistenceReportTemplate,
        "description": (
            "Persistence tracking report showing year-over-year savings "
            "trends with persistence factors, degradation analysis with "
            "rate calculation and half-life projection, equipment "
            "performance decay tracking, operational changes impact, "
            "model stability assessment, and re-commissioning "
            "recommendations with recovery estimates."
        ),
        "category": "tracking",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "2-4 page executive summary showing verified savings "
            "highlights with confidence bounds, financial impact "
            "assessment with ROI and NPV, compliance status traffic "
            "lights across standards, performance trend indicators, "
            "risk assessment, ECM portfolio overview, outlook with "
            "projections, and prioritized action items."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
    {
        "name": "compliance_report",
        "class": ComplianceReportTemplate,
        "description": (
            "Standards compliance report showing IPMVP protocol "
            "checklist with item-level conformity, ISO 50015 clause "
            "assessment, FEMP M&V requirements verification, ASHRAE "
            "Guideline 14 statistical criteria pass/fail, EU EED "
            "Article 7 compliance mapping, data quality assessment, "
            "documentation review, and non-conformity register."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": "40.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-040 M&V report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 M&V report templates. Templates can be listed, filtered
    by category, retrieved by name, and rendered in markdown/HTML/JSON
    formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("mv_plan_report")
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
            name: Template name (e.g., 'mv_plan_report').
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
            category: Category string (e.g., 'planning', 'analysis',
                      'verification', 'reporting', 'tracking',
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
