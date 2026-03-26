# -*- coding: utf-8 -*-
"""
PACK-043 Scope 3 Complete Pack - Report Templates
===================================================

This package provides 10 report templates for the PACK-043 Scope 3
Complete Pack. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256 provenance
hashing for audit trail integrity.

Templates:
    1. EnterpriseDashboardTemplate        - Executive multi-year Scope 3 dashboard
    2. MaturityReportTemplate             - Maturity tier heatmap and upgrade roadmap
    3. LCAProductReportTemplate           - Product carbon footprint with BOM breakdown
    4. ScenarioReportTemplate             - MACC curve and what-if scenario analysis
    5. SBTiProgressReportTemplate         - SBTi target vs actual trajectory tracking
    6. SupplierProgrammeReportTemplate    - Supplier scorecard and commitment tracker
    7. ClimateRiskReportTemplate          - TCFD-aligned transition and physical risk
    8. MultiYearTrendReportTemplate       - Base year comparison and trend decomposition
    9. AssurancePackageTemplate           - ISAE 3410 evidence bundle for verifiers
    10. SectorDisclosureTemplate          - Sector-specific Scope 3 disclosure

Usage:
    >>> from packs.ghg_accounting.PACK_043_scope_3_complete.templates import (
    ...     TemplateRegistry,
    ...     EnterpriseDashboardTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("enterprise_dashboard")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 43.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

_MODULE_VERSION = "43.0.0"

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .enterprise_dashboard import EnterpriseDashboardTemplate
except ImportError as e:
    logger.warning("Failed to import EnterpriseDashboardTemplate: %s", e)
    EnterpriseDashboardTemplate = None  # type: ignore[assignment,misc]

try:
    from .maturity_report import MaturityReportTemplate
except ImportError as e:
    logger.warning("Failed to import MaturityReportTemplate: %s", e)
    MaturityReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .lca_product_report import LCAProductReportTemplate
except ImportError as e:
    logger.warning("Failed to import LCAProductReportTemplate: %s", e)
    LCAProductReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .scenario_report import ScenarioReportTemplate
except ImportError as e:
    logger.warning("Failed to import ScenarioReportTemplate: %s", e)
    ScenarioReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .sbti_progress_report import SBTiProgressReportTemplate
except ImportError as e:
    logger.warning("Failed to import SBTiProgressReportTemplate: %s", e)
    SBTiProgressReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .supplier_programme_report import SupplierProgrammeReportTemplate
except ImportError as e:
    logger.warning("Failed to import SupplierProgrammeReportTemplate: %s", e)
    SupplierProgrammeReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .climate_risk_report import ClimateRiskReportTemplate
except ImportError as e:
    logger.warning("Failed to import ClimateRiskReportTemplate: %s", e)
    ClimateRiskReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .multi_year_trend_report import MultiYearTrendReportTemplate
except ImportError as e:
    logger.warning("Failed to import MultiYearTrendReportTemplate: %s", e)
    MultiYearTrendReportTemplate = None  # type: ignore[assignment,misc]

try:
    from .assurance_package import AssurancePackageTemplate
except ImportError as e:
    logger.warning("Failed to import AssurancePackageTemplate: %s", e)
    AssurancePackageTemplate = None  # type: ignore[assignment,misc]

try:
    from .sector_disclosure import SectorDisclosureTemplate
except ImportError as e:
    logger.warning("Failed to import SectorDisclosureTemplate: %s", e)
    SectorDisclosureTemplate = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "EnterpriseDashboardTemplate",
    "MaturityReportTemplate",
    "LCAProductReportTemplate",
    "ScenarioReportTemplate",
    "SBTiProgressReportTemplate",
    "SupplierProgrammeReportTemplate",
    "ClimateRiskReportTemplate",
    "MultiYearTrendReportTemplate",
    "AssurancePackageTemplate",
    "SectorDisclosureTemplate",
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
    Type[EnterpriseDashboardTemplate],
    Type[MaturityReportTemplate],
    Type[LCAProductReportTemplate],
    Type[ScenarioReportTemplate],
    Type[SBTiProgressReportTemplate],
    Type[SupplierProgrammeReportTemplate],
    Type[ClimateRiskReportTemplate],
    Type[MultiYearTrendReportTemplate],
    Type[AssurancePackageTemplate],
    Type[SectorDisclosureTemplate],
]

TemplateInstance = Union[
    EnterpriseDashboardTemplate,
    MaturityReportTemplate,
    LCAProductReportTemplate,
    ScenarioReportTemplate,
    SBTiProgressReportTemplate,
    SupplierProgrammeReportTemplate,
    ClimateRiskReportTemplate,
    MultiYearTrendReportTemplate,
    AssurancePackageTemplate,
    SectorDisclosureTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "enterprise_dashboard",
        "class": EnterpriseDashboardTemplate,
        "description": (
            "Executive dashboard with Scope 3 multi-year trends, maturity "
            "progress gauge, SBTi trajectory vs target, climate risk summary, "
            "supplier programme status, and top 5 category breakdown. "
            "Designed for C-suite and board-level consumption."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "maturity_report",
        "class": MaturityReportTemplate,
        "description": (
            "Current vs target maturity tier per category heatmap, upgrade "
            "roadmap timeline, ROI analysis per upgrade, uncertainty "
            "reduction forecast, and budget allocation chart. Tracks "
            "progression from spend-based to supplier-specific approaches."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "lca_product_report",
        "class": LCAProductReportTemplate,
        "description": (
            "Product carbon footprint with bill-of-materials breakdown, "
            "lifecycle stage waterfall (raw material, manufacturing, "
            "distribution, use, end-of-life), product comparison table, "
            "sensitivity analysis on key parameters, and circular economy "
            "metrics for product-level decarbonization."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "scenario_report",
        "class": ScenarioReportTemplate,
        "description": (
            "MACC curve data (cost vs reduction per intervention), what-if "
            "scenario results table, cumulative reduction waterfall, Paris "
            "alignment check (1.5C / WB2C), portfolio optimization results, "
            "and budget-constrained top interventions for strategic planning."
        ),
        "category": "strategy",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "sbti_progress_report",
        "class": SBTiProgressReportTemplate,
        "description": (
            "Target vs actual trajectory line chart data, category coverage "
            "assessment, near-term and long-term milestone tracking, FLAG "
            "pathway (if applicable), variance analysis, and submission "
            "package readiness status for SBTi compliance."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "supplier_programme_report",
        "class": SupplierProgrammeReportTemplate,
        "description": (
            "Supplier scorecard summary, commitment tracker (SBTi/RE100/CDP/"
            "net-zero), YoY reduction progress per supplier tier (strategic/"
            "key/managed), programme ROI, and transition risk heatmap for "
            "supply chain decarbonization management."
        ),
        "category": "engagement",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "climate_risk_report",
        "class": ClimateRiskReportTemplate,
        "description": (
            "TCFD-aligned climate risk report: transition risk (carbon pricing, "
            "stranded assets, market shift), physical risk (supply chain "
            "disruption map), opportunities (low-carbon demand), financial "
            "impact NPV table over 10/20/30yr, and scenario comparison "
            "(IEA NZE, NGFS)."
        ),
        "category": "risk",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "multi_year_trend_report",
        "class": MultiYearTrendReportTemplate,
        "description": (
            "Base year vs subsequent years comparison table, recalculation "
            "history log, methodology-adjusted trend line, real reduction "
            "vs methodology change decomposition, and cumulative reduction "
            "since base year for performance tracking."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "assurance_package",
        "class": AssurancePackageTemplate,
        "description": (
            "ISAE 3410 evidence bundle: methodology summary per category, "
            "data source inventory with provenance, calculation log with "
            "SHA-256 hash chain, assumption register, emission factor registry, "
            "completeness statement, uncertainty statement, assurance readiness "
            "score gauge, and verifier query log."
        ),
        "category": "verification",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
    {
        "name": "sector_disclosure",
        "class": SectorDisclosureTemplate,
        "description": (
            "Sector-specific Scope 3 reporting: PCAF financed emissions "
            "disclosure (finance), packaging/logistics analysis (retail), "
            "circular economy metrics (manufacturing), and cloud carbon "
            "report (technology). Parameterized by sector_focus configuration."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json"],
        "version": _MODULE_VERSION,
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-043 Scope 3 Complete Pack report templates.

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
        >>> template = registry.get("enterprise_dashboard")
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
            name: Template name (e.g., 'enterprise_dashboard').
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
            category: Category string (e.g., 'executive', 'analysis',
                      'strategy', 'compliance', 'engagement', 'risk',
                      'verification', 'disclosure').

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
