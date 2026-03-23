# -*- coding: utf-8 -*-
"""
PACK-037 Demand Response Pack - Report Templates
=====================================================

This package provides 10 demand response report templates for the
PACK-037 Demand Response Pack. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates
include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. FlexibilityProfileReportTemplate        - Load flexibility assessment
    2. ProgramAnalysisReportTemplate           - DR program comparison
    3. BaselineAnalysisReportTemplate          - CBL methodology comparison
    4. DispatchPlanReportTemplate              - Event curtailment sequence
    5. EventPerformanceReportTemplate          - Post-event analysis
    6. RevenueDashboardTemplate                - Revenue tracking dashboard
    7. CarbonImpactReportTemplate              - Carbon benefits of DR
    8. DERPerformanceReportTemplate            - DER asset performance
    9. ExecutiveSummaryReportTemplate           - C-suite summary
    10. SettlementVerificationReportTemplate   - Settlement-grade documentation

Usage:
    >>> from packs.energy_efficiency.PACK_037_demand_response.templates import (
    ...     TemplateRegistry,
    ...     FlexibilityProfileReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("flexibility_profile_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 37.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .flexibility_profile_report import FlexibilityProfileReportTemplate
from .program_analysis_report import ProgramAnalysisReportTemplate
from .baseline_analysis_report import BaselineAnalysisReportTemplate
from .dispatch_plan_report import DispatchPlanReportTemplate
from .event_performance_report import EventPerformanceReportTemplate
from .revenue_dashboard import RevenueDashboardTemplate
from .carbon_impact_report import CarbonImpactReportTemplate
from .der_performance_report import DERPerformanceReportTemplate
from .executive_summary_report import ExecutiveSummaryReportTemplate
from .settlement_verification_report import SettlementVerificationReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "FlexibilityProfileReportTemplate",
    "ProgramAnalysisReportTemplate",
    "BaselineAnalysisReportTemplate",
    "DispatchPlanReportTemplate",
    "EventPerformanceReportTemplate",
    "RevenueDashboardTemplate",
    "CarbonImpactReportTemplate",
    "DERPerformanceReportTemplate",
    "ExecutiveSummaryReportTemplate",
    "SettlementVerificationReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[FlexibilityProfileReportTemplate],
    Type[ProgramAnalysisReportTemplate],
    Type[BaselineAnalysisReportTemplate],
    Type[DispatchPlanReportTemplate],
    Type[EventPerformanceReportTemplate],
    Type[RevenueDashboardTemplate],
    Type[CarbonImpactReportTemplate],
    Type[DERPerformanceReportTemplate],
    Type[ExecutiveSummaryReportTemplate],
    Type[SettlementVerificationReportTemplate],
]

TemplateInstance = Union[
    FlexibilityProfileReportTemplate,
    ProgramAnalysisReportTemplate,
    BaselineAnalysisReportTemplate,
    DispatchPlanReportTemplate,
    EventPerformanceReportTemplate,
    RevenueDashboardTemplate,
    CarbonImpactReportTemplate,
    DERPerformanceReportTemplate,
    ExecutiveSummaryReportTemplate,
    SettlementVerificationReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "flexibility_profile_report",
        "class": FlexibilityProfileReportTemplate,
        "description": (
            "Load flexibility assessment report showing all facility loads "
            "categorized by curtailability (fully/partially/non-curtailable), "
            "total curtailment capacity by notification time and duration, "
            "operational constraints, and enrollment recommendations."
        ),
        "category": "assessment",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "program_analysis_report",
        "class": ProgramAnalysisReportTemplate,
        "description": (
            "DR program comparison report with eligibility assessment, "
            "revenue projections (capacity/energy/ancillary), penalty risk "
            "analysis, program comparison matrix with scoring, and "
            "enrollment recommendations."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "baseline_analysis_report",
        "class": BaselineAnalysisReportTemplate,
        "description": (
            "Customer Baseline Load (CBL) methodology comparison showing "
            "projected baselines under each methodology (10-of-10, High-5-of-10, "
            "weather-adjusted regression), historical load profiles, adjustment "
            "factors, and optimization opportunities."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "dispatch_plan_report",
        "class": DispatchPlanReportTemplate,
        "description": (
            "DR event dispatch plan with load curtailment sequence, timing, "
            "kW reduction targets, DER dispatch orders, pre-conditioning "
            "schedule, communication protocol, and contingency actions."
        ),
        "category": "operations",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "event_performance_report",
        "class": EventPerformanceReportTemplate,
        "description": (
            "Post-event performance analysis with actual vs baseline "
            "curtailment comparison, revenue earned and penalties incurred, "
            "load-level performance breakdown, DER contribution analysis, "
            "and lessons learned for continuous improvement."
        ),
        "category": "performance",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "revenue_dashboard",
        "class": RevenueDashboardTemplate,
        "description": (
            "Revenue tracking dashboard across all DR programs with "
            "capacity, energy, and ancillary payment breakdowns, monthly "
            "trends, penalty tracking, program-by-program analysis, and "
            "revenue forecast with confidence bands."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "carbon_impact_report",
        "class": CarbonImpactReportTemplate,
        "description": (
            "Carbon benefits of DR using marginal emission factors with "
            "event-level carbon avoidance, avoided generation mix analysis, "
            "SBTi alignment assessment, and CSRD/ESRS E1 reporting "
            "integration for regulatory disclosure."
        ),
        "category": "carbon",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "der_performance_report",
        "class": DERPerformanceReportTemplate,
        "description": (
            "DER asset performance during DR events covering battery SOC "
            "profiles, solar PV contribution, EV fleet flexibility, thermal "
            "storage utilization, cross-asset coordination effectiveness, "
            "and DER optimization recommendations."
        ),
        "category": "performance",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "2-4 page C-suite summary with total DR revenue, carbon impact, "
            "program compliance scorecard, strategic recommendations, "
            "year-over-year comparison, and key performance indicators "
            "for board-level presentation."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
    {
        "name": "settlement_verification_report",
        "class": SettlementVerificationReportTemplate,
        "description": (
            "Settlement-grade documentation for program administrators "
            "with baseline calculation documentation, interval meter data, "
            "performance verification, adjustment documentation, settlement "
            "calculations, and certification/attestation."
        ),
        "category": "settlement",
        "formats": ["markdown", "html", "json"],
        "version": "37.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-037 Demand Response report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 demand response report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("flexibility_profile_report")
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
            name: Template name (e.g., 'flexibility_profile_report').
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
            category: Category string (e.g., 'assessment', 'analysis',
                      'operations', 'performance', 'dashboard', 'carbon',
                      'executive', 'settlement').

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
