# -*- coding: utf-8 -*-
"""
PACK-041 Scope 1-2 Complete Pack - Report Templates
====================================================

This package provides 10 report templates for the PACK-041 Scope 1-2
Complete Pack. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256 provenance
hashing for audit trail integrity.

Templates:
    1. GHGInventoryReportTemplate           - Complete GHG inventory report
    2. Scope1DetailedReportTemplate         - Detailed Scope 1 per-category
    3. Scope2DualReportTemplate             - Scope 2 dual-method report
    4. EmissionFactorRegistryTemplate       - Complete EF registry
    5. UncertaintyAnalysisReportTemplate    - Uncertainty analysis
    6. TrendAnalysisReportTemplate          - Year-over-year trend analysis
    7. VerificationPackageTemplate          - ISO 14064-3 verification package
    8. ExecutiveSummaryReportTemplate       - 2-4 page executive summary
    9. ComplianceDashboardTemplate          - Multi-framework compliance
    10. ESRSE1DisclosureTemplate            - ESRS E1 climate change disclosure

Usage:
    >>> from packs.ghg_accounting.PACK_041_scope_1_2_complete.templates import (
    ...     TemplateRegistry,
    ...     GHGInventoryReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("ghg_inventory_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 41.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .ghg_inventory_report import GHGInventoryReportTemplate
except ImportError as e:
    logger.warning("Failed to import GHGInventoryReportTemplate: %s", e)
    GHGInventoryReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .scope1_detailed_report import Scope1DetailedReportTemplate
except ImportError as e:
    logger.warning("Failed to import Scope1DetailedReportTemplate: %s", e)
    Scope1DetailedReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .scope2_dual_report import Scope2DualReportTemplate
except ImportError as e:
    logger.warning("Failed to import Scope2DualReportTemplate: %s", e)
    Scope2DualReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .emission_factor_registry import EmissionFactorRegistryTemplate
except ImportError as e:
    logger.warning("Failed to import EmissionFactorRegistryTemplate: %s", e)
    EmissionFactorRegistryTemplate = None  # type: ignore[assignment,misc]

try:
    from .uncertainty_analysis_report import UncertaintyAnalysisReportTemplate
except ImportError as e:
    logger.warning("Failed to import UncertaintyAnalysisReportTemplate: %s", e)
    UncertaintyAnalysisReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .trend_analysis_report import TrendAnalysisReportTemplate
except ImportError as e:
    logger.warning("Failed to import TrendAnalysisReportTemplate: %s", e)
    TrendAnalysisReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .verification_package import VerificationPackageTemplate
except ImportError as e:
    logger.warning("Failed to import VerificationPackageTemplate: %s", e)
    VerificationPackageTemplate = None  # type: ignore[assignment,misc]

try:
    from .executive_summary_report import ExecutiveSummaryReportTemplate
except ImportError as e:
    logger.warning("Failed to import ExecutiveSummaryReportTemplate: %s", e)
    ExecutiveSummaryReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .compliance_dashboard import ComplianceDashboardTemplate
except ImportError as e:
    logger.warning("Failed to import ComplianceDashboardTemplate: %s", e)
    ComplianceDashboardTemplate = None  # type: ignore[assignment,misc]

try:
    from .esrs_e1_disclosure import ESRSE1DisclosureTemplate
except ImportError as e:
    logger.warning("Failed to import ESRSE1DisclosureTemplate: %s", e)
    ESRSE1DisclosureTemplate = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "GHGInventoryReportTemplate",
    "Scope1DetailedReportTemplate",
    "Scope2DualReportTemplate",
    "EmissionFactorRegistryTemplate",
    "UncertaintyAnalysisReportTemplate",
    "TrendAnalysisReportTemplate",
    "VerificationPackageTemplate",
    "ExecutiveSummaryReportTemplate",
    "ComplianceDashboardTemplate",
    "ESRSE1DisclosureTemplate",
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
    Type[GHGInventoryReportTemplate],
    Type[Scope1DetailedReportTemplate],
    Type[Scope2DualReportTemplate],
    Type[EmissionFactorRegistryTemplate],
    Type[UncertaintyAnalysisReportTemplate],
    Type[TrendAnalysisReportTemplate],
    Type[VerificationPackageTemplate],
    Type[ExecutiveSummaryReportTemplate],
    Type[ComplianceDashboardTemplate],
    Type[ESRSE1DisclosureTemplate],
]

TemplateInstance = Union[
    GHGInventoryReportTemplate,
    Scope1DetailedReportTemplate,
    Scope2DualReportTemplate,
    EmissionFactorRegistryTemplate,
    UncertaintyAnalysisReportTemplate,
    TrendAnalysisReportTemplate,
    VerificationPackageTemplate,
    ExecutiveSummaryReportTemplate,
    ComplianceDashboardTemplate,
    ESRSE1DisclosureTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "ghg_inventory_report",
        "class": GHGInventoryReportTemplate,
        "description": (
            "Complete GHG inventory report with executive summary, "
            "organizational boundary, Scope 1 breakdown (8 categories, "
            "7 gases, per facility, per entity), Scope 2 dual reporting "
            "(location + market), combined totals, YoY comparison, "
            "uncertainty summary, methodology notes, completeness "
            "statement, and data quality assessment."
        ),
        "category": "inventory",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "scope1_detailed_report",
        "class": Scope1DetailedReportTemplate,
        "description": (
            "Detailed Scope 1 per-category report with activity data, "
            "emission factors, per-gas emissions, equipment/asset "
            "inventory for stationary/mobile/refrigerant, process "
            "methodologies, fugitive estimation methods, agricultural "
            "factors, cross-category reconciliation, and EF citations."
        ),
        "category": "scope1",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "scope2_dual_report",
        "class": Scope2DualReportTemplate,
        "description": (
            "Scope 2 dual-method report with location-based by facility "
            "with grid EFs, market-based with instrument allocation, "
            "steam/heat/cooling by supplier, instrument portfolio "
            "(PPAs, RECs, GOs), residual mix application, location vs "
            "market variance analysis, RE procurement impact, and "
            "quality criteria compliance."
        ),
        "category": "scope2",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "emission_factor_registry",
        "class": EmissionFactorRegistryTemplate,
        "description": (
            "Complete emission factor registry with summary statistics, "
            "per-source-category factor tables (value, unit, source, "
            "version, geography, year, provenance hash), GWP values "
            "(AR4/5/6), factor overrides with justification, consistency "
            "checks, and data quality assessment."
        ),
        "category": "registry",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "uncertainty_analysis_report",
        "class": UncertaintyAnalysisReportTemplate,
        "description": (
            "Uncertainty analysis with methodology description "
            "(analytical + Monte Carlo), per-source inputs, analytical "
            "results (combined uncertainty, CI), Monte Carlo results "
            "(distribution histogram, percentiles), top contributors "
            "ranked, sensitivity analysis, and improvement recommendations."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "trend_analysis_report",
        "class": TrendAnalysisReportTemplate,
        "description": (
            "Year-over-year trend analysis with multi-year emission "
            "summary, absolute/percentage changes, Kaya decomposition "
            "chart data, per-factor contributions, intensity metrics "
            "(tCO2e/revenue, /FTE, /m2), SBTi trajectory comparison, "
            "base year recalculation history, and weather normalization."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "verification_package",
        "class": VerificationPackageTemplate,
        "description": (
            "ISO 14064-3 verification package with organization "
            "description, inventory summary, methodology per category, "
            "EF provenance (SHA-256), activity data evidence, "
            "uncertainty analysis, base year recalculations, "
            "completeness statement, full calculation audit trail "
            "with SHA-256 chain, and quality management procedures."
        ),
        "category": "verification",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "2-4 page executive summary with headline metrics "
            "(Scope 1, Scope 2 location/market), key changes from "
            "prior year, top 5 emission sources, RE procurement impact, "
            "SBTi progress, compliance status across frameworks, "
            "recommended actions, and infographic-ready data points."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "compliance_dashboard",
        "class": ComplianceDashboardTemplate,
        "description": (
            "Multi-framework compliance dashboard with readiness scores "
            "(0-100 bar chart data), per-framework gap analysis, "
            "critical gaps requiring immediate action, remediation "
            "action plan with priority, submission timeline/deadlines, "
            "and framework-specific notes."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
    {
        "name": "esrs_e1_disclosure",
        "class": ESRSE1DisclosureTemplate,
        "description": (
            "ESRS E1 Climate Change disclosure with E1-1 transition "
            "plan, E1-4 emission reduction targets, E1-5 energy "
            "consumption and mix, E1-6 Scope 1/2/3 emissions in "
            "EFRAG format, biogenic CO2 separate reporting, ESRS "
            "methodology notes, and XBRL-ready data structure in "
            "JSON output."
        ),
        "category": "esrs",
        "formats": ["markdown", "html", "json"],
        "version": "41.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-041 Scope 1-2 Complete report templates.

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
        >>> template = registry.get("ghg_inventory_report")
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

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'ghg_inventory_report').
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
            category: Category string (e.g., 'inventory', 'scope1',
                      'scope2', 'registry', 'analysis', 'verification',
                      'executive', 'compliance', 'esrs').

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
