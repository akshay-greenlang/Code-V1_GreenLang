# -*- coding: utf-8 -*-
"""
PACK-042 Scope 3 Starter Pack - Report Templates
==================================================

This package provides 10 report templates for the PACK-042 Scope 3
Starter Pack. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256 provenance
hashing for audit trail integrity.

Templates:
    1. Scope3InventoryReportTemplate        - Full 15-category Scope 3 inventory
    2. CategoryDeepDiveReportTemplate       - Single-category deep dive analysis
    3. Scope3ExecutiveSummaryTemplate       - C-suite executive summary
    4. HotspotReportTemplate               - Pareto analysis and prioritization
    5. SupplierEngagementReportTemplate     - Supplier engagement dashboard
    6. DataQualityReportTemplate            - Data quality assessment and roadmap
    7. Scope3ComplianceDashboardTemplate    - Multi-framework compliance readiness
    8. Scope3UncertaintyReportTemplate      - Monte Carlo and sensitivity analysis
    9. Scope3VerificationPackageTemplate    - ISO 14064-3 / ISAE 3410 evidence
    10. ESRSE1Scope3DisclosureTemplate      - ESRS E1-6 Scope 3 disclosure

Usage:
    >>> from packs.ghg_accounting.PACK_042_scope_3_starter.templates import (
    ...     TemplateRegistry,
    ...     Scope3InventoryReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("scope3_inventory_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 42.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

_MODULE_VERSION = "42.0.0"

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .scope3_inventory_report import Scope3InventoryReportTemplate
except ImportError as e:
    logger.warning("Failed to import Scope3InventoryReportTemplate: %s", e)
    Scope3InventoryReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .category_deep_dive_report import CategoryDeepDiveReportTemplate
except ImportError as e:
    logger.warning("Failed to import CategoryDeepDiveReportTemplate: %s", e)
    CategoryDeepDiveReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .scope3_executive_summary import Scope3ExecutiveSummaryTemplate
except ImportError as e:
    logger.warning("Failed to import Scope3ExecutiveSummaryTemplate: %s", e)
    Scope3ExecutiveSummaryTemplate = None  # type: ignore[assignment,misc]

try:
    from .hotspot_report import HotspotReportTemplate
except ImportError as e:
    logger.warning("Failed to import HotspotReportTemplate: %s", e)
    HotspotReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .supplier_engagement_report import SupplierEngagementReportTemplate
except ImportError as e:
    logger.warning("Failed to import SupplierEngagementReportTemplate: %s", e)
    SupplierEngagementReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .data_quality_report import DataQualityReportTemplate
except ImportError as e:
    logger.warning("Failed to import DataQualityReportTemplate: %s", e)
    DataQualityReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .scope3_compliance_dashboard import Scope3ComplianceDashboardTemplate
except ImportError as e:
    logger.warning("Failed to import Scope3ComplianceDashboardTemplate: %s", e)
    Scope3ComplianceDashboardTemplate = None  # type: ignore[assignment,misc]

try:
    from .scope3_uncertainty_report import Scope3UncertaintyReportTemplate
except ImportError as e:
    logger.warning("Failed to import Scope3UncertaintyReportTemplate: %s", e)
    Scope3UncertaintyReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .scope3_verification_package import Scope3VerificationPackageTemplate
except ImportError as e:
    logger.warning("Failed to import Scope3VerificationPackageTemplate: %s", e)
    Scope3VerificationPackageTemplate = None  # type: ignore[assignment,misc]

try:
    from .esrs_e1_scope3_disclosure import ESRSE1Scope3DisclosureTemplate
except ImportError as e:
    logger.warning("Failed to import ESRSE1Scope3DisclosureTemplate: %s", e)
    ESRSE1Scope3DisclosureTemplate = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "Scope3InventoryReportTemplate",
    "CategoryDeepDiveReportTemplate",
    "Scope3ExecutiveSummaryTemplate",
    "HotspotReportTemplate",
    "SupplierEngagementReportTemplate",
    "DataQualityReportTemplate",
    "Scope3ComplianceDashboardTemplate",
    "Scope3UncertaintyReportTemplate",
    "Scope3VerificationPackageTemplate",
    "ESRSE1Scope3DisclosureTemplate",
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
    Type[Scope3InventoryReportTemplate],
    Type[CategoryDeepDiveReportTemplate],
    Type[Scope3ExecutiveSummaryTemplate],
    Type[HotspotReportTemplate],
    Type[SupplierEngagementReportTemplate],
    Type[DataQualityReportTemplate],
    Type[Scope3ComplianceDashboardTemplate],
    Type[Scope3UncertaintyReportTemplate],
    Type[Scope3VerificationPackageTemplate],
    Type[ESRSE1Scope3DisclosureTemplate],
]

TemplateInstance = Union[
    Scope3InventoryReportTemplate,
    CategoryDeepDiveReportTemplate,
    Scope3ExecutiveSummaryTemplate,
    HotspotReportTemplate,
    SupplierEngagementReportTemplate,
    DataQualityReportTemplate,
    Scope3ComplianceDashboardTemplate,
    Scope3UncertaintyReportTemplate,
    Scope3VerificationPackageTemplate,
    ESRSE1Scope3DisclosureTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "scope3_inventory_report",
        "class": Scope3InventoryReportTemplate,
        "description": (
            "Full 15-category Scope 3 inventory report with executive "
            "summary, methodology overview, per-category breakdown table "
            "(tCO2e, % of total, methodology tier, data quality, YoY "
            "change), upstream vs downstream split, gas-level breakdown, "
            "data quality summary, uncertainty ranges, compliance status, "
            "and appendix with EF sources and assumptions."
        ),
        "category": "inventory",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "category_deep_dive_report",
        "class": CategoryDeepDiveReportTemplate,
        "description": (
            "Detailed single-category analysis parameterized for any of "
            "15 Scope 3 categories with sub-category breakdown, emission "
            "factor sources, supplier/product contributions, methodology "
            "description, data quality assessment, uncertainty range, "
            "and reduction opportunities."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "scope3_executive_summary",
        "class": Scope3ExecutiveSummaryTemplate,
        "description": (
            "2-4 page C-suite executive summary with total Scope 3 "
            "headline number, top 5 categories waterfall chart data, "
            "% of total footprint (Scope 1+2+3), YoY trend, SBTi "
            "alignment status, 3 priority actions, and data quality "
            "summary designed for board/executive consumption."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "hotspot_report",
        "class": HotspotReportTemplate,
        "description": (
            "Pareto analysis and prioritization report with categories "
            "ranked by contribution, materiality matrix data, supplier "
            "concentration analysis, geographic distribution, product "
            "carbon intensity ranking, reduction opportunities with ROI, "
            "and tier upgrade impact quantification."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "supplier_engagement_report",
        "class": SupplierEngagementReportTemplate,
        "description": (
            "Supplier engagement status dashboard with overall engagement "
            "metrics, response rate trends, data quality distribution "
            "(Level 1-5), top supplier profiles, engagement ROI, "
            "upcoming deadlines, and quality improvement trajectory."
        ),
        "category": "engagement",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "data_quality_report",
        "class": DataQualityReportTemplate,
        "description": (
            "Data quality assessment and improvement roadmap with overall "
            "DQR score, per-category quality spider/radar chart data, "
            "5 DQI breakdown per category, quality trend over time, gap "
            "analysis, prioritized improvement actions with effort/impact "
            "scoring, and framework minimum thresholds comparison."
        ),
        "category": "quality",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "scope3_compliance_dashboard",
        "class": Scope3ComplianceDashboardTemplate,
        "description": (
            "Multi-framework compliance readiness dashboard with per-"
            "framework compliance scores (0-100%), requirements checklist "
            "with pass/fail, gap analysis summary, action items with "
            "priority and effort, cross-framework coverage matrix for "
            "GHG Protocol, ESRS E1, CDP, SBTi, SEC, and SB 253."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "scope3_uncertainty_report",
        "class": Scope3UncertaintyReportTemplate,
        "description": (
            "Monte Carlo results and sensitivity analysis with total "
            "Scope 3 95% CI, per-category uncertainty ranges, probability "
            "distribution histogram data, sensitivity tornado chart data, "
            "methodology tier vs uncertainty correlation, tier upgrade "
            "impact quantification, and correlation matrix."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "scope3_verification_package",
        "class": Scope3VerificationPackageTemplate,
        "description": (
            "ISO 14064-3 / ISAE 3410 evidence bundle with methodology "
            "summary per category, data source inventory, emission factor "
            "registry with provenance, calculation log with SHA-256 "
            "hashes, assumption register, materiality assessment, "
            "completeness statement, data quality statement, uncertainty "
            "statement, and organization boundary reference."
        ),
        "category": "verification",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "esrs_e1_scope3_disclosure",
        "class": ESRSE1Scope3DisclosureTemplate,
        "description": (
            "ESRS E1-6 para 44-46 formatted Scope 3 disclosure with "
            "total and per-category emissions, methodology description "
            "per ESRS requirements, data quality statement, phase-in "
            "compliance status (2025 Cat 1-3, 2029 all), XBRL data "
            "points, significant category rationale, and estimation "
            "methodology and assumptions."
        ),
        "category": "esrs",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-042 Scope 3 Starter Pack report templates.

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
        >>> template = registry.get("scope3_inventory_report")
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
            name: Template name (e.g., 'scope3_inventory_report').
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
            category: Category string (e.g., 'inventory', 'analysis',
                      'executive', 'engagement', 'quality', 'compliance',
                      'verification', 'esrs').

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
