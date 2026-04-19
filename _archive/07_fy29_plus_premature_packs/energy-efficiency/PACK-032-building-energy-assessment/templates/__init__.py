# -*- coding: utf-8 -*-
"""
PACK-032 Building Energy Assessment Pack - Report Templates
=============================================================

This package provides 10 building energy assessment report templates
for the PACK-032 Building Energy Assessment Pack. Each template supports
three rendering formats: Markdown, HTML (with inline CSS), and JSON.
All templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1.  EPCReportTemplate                       - Energy Performance Certificate (A-G)
    2.  DECReportTemplate                       - Display Energy Certificate (0-150+)
    3.  BuildingAssessmentReportTemplate         - Comprehensive building assessment
    4.  RetrofitRecommendationReportTemplate     - Retrofit business case with NPV/IRR
    5.  BuildingBenchmarkReportTemplate          - Peer comparison and benchmarking
    6.  CertificationScorecardTemplate           - LEED/BREEAM/Energy Star scorecard
    7.  TenantEnergyReportTemplate               - Tenant-facing energy report
    8.  BuildingDashboardTemplate                - Real-time energy dashboard
    9.  RegulatoryComplianceReportTemplate       - EPBD/MEES compliance report
    10. WholeLifeCarbonReportTemplate            - Embodied + operational carbon (EN 15978)

Usage:
    >>> from packs.energy_efficiency.PACK_032_building_energy_assessment.templates import (
    ...     TemplateRegistry,
    ...     EPCReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("epc_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Filter by category
    >>> cert_templates = registry.get_by_category("certification")

Author: GreenLang Team
Version: 32.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .epc_report import EPCReportTemplate
from .dec_report import DECReportTemplate
from .building_assessment_report import BuildingAssessmentReportTemplate
from .retrofit_recommendation_report import RetrofitRecommendationReportTemplate
from .building_benchmark_report import BuildingBenchmarkReportTemplate
from .certification_scorecard import CertificationScorecardTemplate
from .tenant_energy_report import TenantEnergyReportTemplate
from .building_dashboard import BuildingDashboardTemplate
from .regulatory_compliance_report import RegulatoryComplianceReportTemplate
from .whole_life_carbon_report import WholeLifeCarbonReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "EPCReportTemplate",
    "DECReportTemplate",
    "BuildingAssessmentReportTemplate",
    "RetrofitRecommendationReportTemplate",
    "BuildingBenchmarkReportTemplate",
    "CertificationScorecardTemplate",
    "TenantEnergyReportTemplate",
    "BuildingDashboardTemplate",
    "RegulatoryComplianceReportTemplate",
    "WholeLifeCarbonReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[EPCReportTemplate],
    Type[DECReportTemplate],
    Type[BuildingAssessmentReportTemplate],
    Type[RetrofitRecommendationReportTemplate],
    Type[BuildingBenchmarkReportTemplate],
    Type[CertificationScorecardTemplate],
    Type[TenantEnergyReportTemplate],
    Type[BuildingDashboardTemplate],
    Type[RegulatoryComplianceReportTemplate],
    Type[WholeLifeCarbonReportTemplate],
]

TemplateInstance = Union[
    EPCReportTemplate,
    DECReportTemplate,
    BuildingAssessmentReportTemplate,
    RetrofitRecommendationReportTemplate,
    BuildingBenchmarkReportTemplate,
    CertificationScorecardTemplate,
    TenantEnergyReportTemplate,
    BuildingDashboardTemplate,
    RegulatoryComplianceReportTemplate,
    WholeLifeCarbonReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "epc_report",
        "class": EPCReportTemplate,
        "description": (
            "Energy Performance Certificate (EPC) report with A-G rating "
            "bands, building details, energy use breakdown, CO2 emissions, "
            "cost-effective and further recommendations, estimated costs, "
            "Green Deal eligibility, and assessor details."
        ),
        "category": "certificate",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "dec_report",
        "class": DECReportTemplate,
        "description": (
            "Display Energy Certificate (DEC) report for public buildings "
            "with operational rating on a 0-150+ scale, electricity/heating/"
            "renewable breakdowns, previous ratings comparison, and advisory "
            "report references."
        ),
        "category": "certificate",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "building_assessment_report",
        "class": BuildingAssessmentReportTemplate,
        "description": (
            "Comprehensive building energy assessment report covering "
            "envelope, HVAC, lighting, DHW, renewables, indoor environment "
            "quality, benchmark comparison, improvement recommendations, "
            "and implementation roadmap."
        ),
        "category": "assessment",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "retrofit_recommendation_report",
        "class": RetrofitRecommendationReportTemplate,
        "description": (
            "Retrofit business case report with current performance baseline, "
            "retrofit measures, financial analysis (NPV/IRR/payback), MACC "
            "curve data, staged implementation roadmap, funding options, "
            "environmental impact, and risk assessment."
        ),
        "category": "retrofit",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "building_benchmark_report",
        "class": BuildingBenchmarkReportTemplate,
        "description": (
            "Building energy benchmark and peer comparison report with EUI "
            "analysis, DEC rating context, CRREM pathway compliance, Energy "
            "Star score, end-use breakdown, weather normalization, peer "
            "comparison chart data, and gap to best practice."
        ),
        "category": "benchmark",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "certification_scorecard",
        "class": CertificationScorecardTemplate,
        "description": (
            "Green building certification scorecard for LEED, BREEAM, "
            "Energy Star, and other schemes. Includes credit category "
            "scores, prerequisites status, gap analysis, action items "
            "for next level, cost estimates, and implementation timeline."
        ),
        "category": "certification",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "tenant_energy_report",
        "class": TenantEnergyReportTemplate,
        "description": (
            "Tenant-facing energy report with energy and cost summaries, "
            "comparison to building average and external benchmarks, "
            "monthly trend data, energy reduction tips, green lease "
            "compliance status, and carbon footprint attribution."
        ),
        "category": "tenant",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "building_dashboard",
        "class": BuildingDashboardTemplate,
        "description": (
            "Real-time building energy dashboard with KPI summary cards, "
            "consumption trends, end-use breakdown, cost tracking, carbon "
            "intensity, weather normalization, alerts and anomalies, "
            "targets vs actuals, and occupancy correlation."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "regulatory_compliance_report",
        "class": RegulatoryComplianceReportTemplate,
        "description": (
            "Building energy regulatory compliance report covering EPBD "
            "recast, MEES, and local regulations. Includes current ratings, "
            "minimum requirements, compliance timeline, penalty exposure, "
            "required improvements, and action plan."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
    {
        "name": "whole_life_carbon_report",
        "class": WholeLifeCarbonReportTemplate,
        "description": (
            "Whole life carbon assessment report per EN 15978 with lifecycle "
            "stage breakdown (A1-D), material carbon hotspots, operational "
            "carbon projection, RIBA/LETI target comparison, material "
            "substitution opportunities, biogenic carbon, and sensitivity "
            "analysis."
        ),
        "category": "carbon",
        "formats": ["markdown", "html", "json"],
        "version": "32.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-032 Building Energy Assessment report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 building energy assessment report templates. Templates can
    be listed, filtered by category, retrieved by name, and rendered
    in markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("epc_report")
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
            name: Template name (e.g., 'epc_report').
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
            category: Category string (e.g., 'certificate', 'assessment',
                      'retrofit', 'benchmark', 'certification', 'tenant',
                      'dashboard', 'compliance', 'carbon').

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
