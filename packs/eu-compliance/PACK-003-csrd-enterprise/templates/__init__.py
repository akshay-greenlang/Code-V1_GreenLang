"""
PACK-003 CSRD Enterprise Pack - Report Templates.

This package provides 9 enterprise report templates for the CSRD Enterprise
Pack. Each template supports three rendering formats: Markdown, HTML (with
inline CSS), and JSON. All templates include SHA-256 provenance hashing
for audit trail integrity.

Templates:
    1. EnterpriseDashboardTemplate - Multi-tenant overview dashboard
    2. WhiteLabelReportTemplate - Branded report generator
    3. PredictiveInsightsTemplate - AI forecast visualization
    4. AuditorPortalViewTemplate - Auditor workspace
    5. SupplyChainReportTemplate - Supply chain ESG scorecard
    6. CarbonCreditReportTemplate - Carbon credit portfolio report
    7. RegulatoryFilingReportTemplate - Filing status and history
    8. ExecutiveCockpitTemplate - C-suite real-time dashboard
    9. CustomReportBuilderTemplate - User-defined report composition

Usage:
    >>> from packs.eu_compliance.PACK_003_csrd_enterprise.templates import (
    ...     TemplateRegistry,
    ...     EnterpriseDashboardTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("enterprise_dashboard")
    >>> html = template.render_html(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .enterprise_dashboard import EnterpriseDashboardTemplate
from .white_label_report import WhiteLabelReportTemplate
from .predictive_insights import PredictiveInsightsTemplate
from .auditor_portal_view import AuditorPortalViewTemplate
from .supply_chain_report import SupplyChainReportTemplate
from .carbon_credit_report import CarbonCreditReportTemplate
from .regulatory_filing_report import RegulatoryFilingReportTemplate
from .executive_cockpit import ExecutiveCockpitTemplate
from .custom_report_builder import (
    CustomReportBuilderTemplate,
    WidgetType,
    WIDGET_CATALOG,
    _make_widget,
    _make_layout,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "EnterpriseDashboardTemplate",
    "WhiteLabelReportTemplate",
    "PredictiveInsightsTemplate",
    "AuditorPortalViewTemplate",
    "SupplyChainReportTemplate",
    "CarbonCreditReportTemplate",
    "RegulatoryFilingReportTemplate",
    "ExecutiveCockpitTemplate",
    "CustomReportBuilderTemplate",
    # Registry
    "TemplateRegistry",
    # Custom report builder utilities
    "WidgetType",
    "WIDGET_CATALOG",
    "make_widget",
    "make_layout",
]

# Public aliases for factory functions
make_widget = _make_widget
make_layout = _make_layout


# Template type alias for registry
TemplateClass = Union[
    Type[EnterpriseDashboardTemplate],
    Type[WhiteLabelReportTemplate],
    Type[PredictiveInsightsTemplate],
    Type[AuditorPortalViewTemplate],
    Type[SupplyChainReportTemplate],
    Type[CarbonCreditReportTemplate],
    Type[RegulatoryFilingReportTemplate],
    Type[ExecutiveCockpitTemplate],
    Type[CustomReportBuilderTemplate],
]

# Template instance type alias
TemplateInstance = Union[
    EnterpriseDashboardTemplate,
    WhiteLabelReportTemplate,
    PredictiveInsightsTemplate,
    AuditorPortalViewTemplate,
    SupplyChainReportTemplate,
    CarbonCreditReportTemplate,
    RegulatoryFilingReportTemplate,
    ExecutiveCockpitTemplate,
    CustomReportBuilderTemplate,
]


class TemplateRegistry:
    """
    Registry for PACK-003 CSRD Enterprise report templates.

    Provides a centralized way to discover, instantiate, and manage
    all 9 enterprise report templates. Templates can be listed, retrieved
    by name, and instantiated with optional configuration.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> template = registry.get("enterprise_dashboard")
        >>> html = template.render_html(data)
    """

    _TEMPLATE_DEFINITIONS: List[Dict[str, Any]] = [
        {
            "name": "enterprise_dashboard",
            "class": EnterpriseDashboardTemplate,
            "description": "Multi-tenant overview dashboard with KPI cards, "
                           "compliance heatmap, emission trends, alerts, "
                           "and tenant health scores.",
            "category": "dashboard",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "white_label_report",
            "class": WhiteLabelReportTemplate,
            "description": "Branded report generator with dynamic header, "
                           "cover page, table of contents, and CSS variable "
                           "injection for tenant branding.",
            "category": "report",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "predictive_insights",
            "class": PredictiveInsightsTemplate,
            "description": "AI forecast visualization with emission forecasts, "
                           "gap-to-target analysis, anomaly detection, feature "
                           "importance, Monte Carlo distributions, and model metrics.",
            "category": "analytics",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "auditor_portal_view",
            "class": AuditorPortalViewTemplate,
            "description": "Auditor workspace with engagement overview, "
                           "evidence browser, finding tracker, threaded "
                           "discussions, and assurance opinion management.",
            "category": "audit",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "supply_chain_report",
            "class": SupplyChainReportTemplate,
            "description": "Supply chain ESG scorecard with supplier map, "
                           "scorecards, risk distribution, sector benchmarks, "
                           "and Scope 3 emission breakdowns.",
            "category": "supply_chain",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "carbon_credit_report",
            "class": CarbonCreditReportTemplate,
            "description": "Carbon credit portfolio report with vintage "
                           "breakdown, quality distribution, net-zero waterfall, "
                           "retirement schedule, and SBTi compliance notes.",
            "category": "carbon",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "regulatory_filing_report",
            "class": RegulatoryFilingReportTemplate,
            "description": "Filing status and history with calendar, submission "
                           "history, validation results, version comparison, "
                           "and filing provenance chain.",
            "category": "regulatory",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "executive_cockpit",
            "class": ExecutiveCockpitTemplate,
            "description": "C-suite real-time dashboard with KPIs, risk radar, "
                           "compliance trajectory, peer benchmarking, board "
                           "actions, financial impact, and sparklines.",
            "category": "executive",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
        {
            "name": "custom_report_builder",
            "class": CustomReportBuilderTemplate,
            "description": "User-defined report composition with 30+ widget "
                           "types, layout management, validation, and "
                           "multi-format rendering.",
            "category": "builder",
            "formats": ["markdown", "html", "json"],
            "version": "1.0.0",
        },
    ]

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, TemplateInstance] = {}

        for defn in self._TEMPLATE_DEFINITIONS:
            self._templates[defn["name"]] = defn

        logger.info(
            "TemplateRegistry initialized with %d templates",
            len(self._templates),
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with metadata.

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
            for defn in self._TEMPLATE_DEFINITIONS
        ]

    def list_template_names(self) -> List[str]:
        """List all available template names.

        Returns:
            List of template name strings.
        """
        return [defn["name"] for defn in self._TEMPLATE_DEFINITIONS]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TemplateInstance:
        """Get a template instance by name.

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'enterprise_dashboard').
            config: Optional configuration overrides.

        Returns:
            Template instance.

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

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a specific template.

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
        """Get templates filtered by category.

        Args:
            category: Category string (e.g., 'dashboard', 'report',
                      'analytics', 'audit', 'supply_chain', 'carbon',
                      'regulatory', 'executive', 'builder').

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
            for defn in self._TEMPLATE_DEFINITIONS
            if defn["category"] == category
        ]

    def has_template(self, name: str) -> bool:
        """Check if a template exists by name.

        Args:
            name: Template name to check.

        Returns:
            True if template exists.
        """
        return name in self._templates

    @property
    def template_count(self) -> int:
        """Return the number of registered templates.

        Returns:
            Template count.
        """
        return len(self._templates)

    def __repr__(self) -> str:
        """Return string representation of registry.

        Returns:
            Repr string.
        """
        return (
            f"TemplateRegistry(templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )
