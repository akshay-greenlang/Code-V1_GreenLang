# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Report Templates
========================================================

This package provides 9 ESRS E1 climate disclosure report templates for
the PACK-016 E1 Climate Pack. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates
include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. GHGEmissionsReportTemplate          - E1-6 GHG emissions disclosure
    2. EnergyMixReportTemplate             - E1-5 energy consumption and mix
    3. TransitionPlanReportTemplate         - E1-1 transition plan for mitigation
    4. ClimatePolicyReportTemplate          - E1-2 climate policies
    5. ClimateActionsReportTemplate         - E1-3 actions and resources
    6. ClimateTargetsReportTemplate         - E1-4 targets and progress
    7. CarbonCreditsReportTemplate          - E1-7 carbon credits and removals
    8. CarbonPricingReportTemplate          - E1-8 internal carbon pricing
    9. ClimateRiskReportTemplate            - E1-9 climate risk and opportunity

Usage:
    >>> from packs.eu_compliance.PACK_016_esrs_e1_climate.templates import (
    ...     TemplateRegistry,
    ...     GHGEmissionsReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("ghg_emissions_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 16.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.ghg_emissions_report import (
        GHGEmissionsReportTemplate,
    )
except ImportError:
    GHGEmissionsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GHGEmissionsReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.energy_mix_report import (
        EnergyMixReportTemplate,
    )
except ImportError:
    EnergyMixReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import EnergyMixReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.transition_plan_report import (
        TransitionPlanReportTemplate,
    )
except ImportError:
    TransitionPlanReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TransitionPlanReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.climate_policy_report import (
        ClimatePolicyReportTemplate,
    )
except ImportError:
    ClimatePolicyReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ClimatePolicyReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.climate_actions_report import (
        ClimateActionsReportTemplate,
    )
except ImportError:
    ClimateActionsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ClimateActionsReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.climate_targets_report import (
        ClimateTargetsReportTemplate,
    )
except ImportError:
    ClimateTargetsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ClimateTargetsReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.carbon_credits_report import (
        CarbonCreditsReportTemplate,
    )
except ImportError:
    CarbonCreditsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CarbonCreditsReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.carbon_pricing_report import (
        CarbonPricingReportTemplate,
    )
except ImportError:
    CarbonPricingReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CarbonPricingReportTemplate")

try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.templates.climate_risk_report import (
        ClimateRiskReportTemplate,
    )
except ImportError:
    ClimateRiskReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ClimateRiskReportTemplate")


__all__ = [
    # Template classes
    "GHGEmissionsReportTemplate",
    "EnergyMixReportTemplate",
    "TransitionPlanReportTemplate",
    "ClimatePolicyReportTemplate",
    "ClimateActionsReportTemplate",
    "ClimateTargetsReportTemplate",
    "CarbonCreditsReportTemplate",
    "CarbonPricingReportTemplate",
    "ClimateRiskReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type aliases (used when all classes are available)
TemplateClass = Union[
    Type["GHGEmissionsReportTemplate"],
    Type["EnergyMixReportTemplate"],
    Type["TransitionPlanReportTemplate"],
    Type["ClimatePolicyReportTemplate"],
    Type["ClimateActionsReportTemplate"],
    Type["ClimateTargetsReportTemplate"],
    Type["CarbonCreditsReportTemplate"],
    Type["CarbonPricingReportTemplate"],
    Type["ClimateRiskReportTemplate"],
]

TemplateInstance = Union[
    "GHGEmissionsReportTemplate",
    "EnergyMixReportTemplate",
    "TransitionPlanReportTemplate",
    "ClimatePolicyReportTemplate",
    "ClimateActionsReportTemplate",
    "ClimateTargetsReportTemplate",
    "CarbonCreditsReportTemplate",
    "CarbonPricingReportTemplate",
    "ClimateRiskReportTemplate",
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "ghg_emissions_report",
        "class": GHGEmissionsReportTemplate,
        "description": (
            "GHG emissions disclosure report with Scope 1/2/3 breakdowns, "
            "gas disaggregation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3), "
            "intensity metrics, and base-year comparison per ESRS E1-6."
        ),
        "category": "emissions",
        "esrs_reference": "E1-6",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "energy_mix_report",
        "class": EnergyMixReportTemplate,
        "description": (
            "Energy consumption and mix report with fossil/renewable breakdown, "
            "self-generated vs purchased energy, intensity metrics, and "
            "renewable share progress tracking per ESRS E1-5."
        ),
        "category": "energy",
        "esrs_reference": "E1-5",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "transition_plan_report",
        "class": TransitionPlanReportTemplate,
        "description": (
            "Climate transition plan report with Paris-aligned scenario analysis, "
            "decarbonization levers, action timeline, CapEx allocation, "
            "locked-in emissions, and gap analysis per ESRS E1-1."
        ),
        "category": "strategy",
        "esrs_reference": "E1-1",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "climate_policy_report",
        "class": ClimatePolicyReportTemplate,
        "description": (
            "Climate policy disclosure with mitigation and adaptation policy "
            "details, governance oversight, value chain coverage, and "
            "framework alignment per ESRS E1-2."
        ),
        "category": "governance",
        "esrs_reference": "E1-2",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "climate_actions_report",
        "class": ClimateActionsReportTemplate,
        "description": (
            "Climate actions and resources report with action summary, "
            "resource allocation, implementation timeline, EU Taxonomy "
            "alignment, and KPI-based progress tracking per ESRS E1-3."
        ),
        "category": "actions",
        "esrs_reference": "E1-3",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "climate_targets_report",
        "class": ClimateTargetsReportTemplate,
        "description": (
            "Climate targets report with SBTi alignment status, progress "
            "tracker, base year information, interim milestones, and "
            "absolute/intensity target breakdown per ESRS E1-4."
        ),
        "category": "targets",
        "esrs_reference": "E1-4",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "carbon_credits_report",
        "class": CarbonCreditsReportTemplate,
        "description": (
            "Carbon credits disclosure with portfolio summary, removal vs "
            "avoidance breakdown, credit quality tiers, project details, "
            "and SBTi BVCM compliance per ESRS E1-7."
        ),
        "category": "credits",
        "esrs_reference": "E1-7",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "carbon_pricing_report",
        "class": CarbonPricingReportTemplate,
        "description": (
            "Carbon pricing disclosure with ETS/tax mechanism overview, "
            "emissions coverage summary, internal shadow pricing for "
            "investment decisions, and scenario analysis per ESRS E1-8."
        ),
        "category": "pricing",
        "esrs_reference": "E1-8",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
    {
        "name": "climate_risk_report",
        "class": ClimateRiskReportTemplate,
        "description": (
            "Climate risk and opportunity report with physical risk "
            "(acute/chronic), transition risk (policy/technology/market), "
            "opportunity analysis, scenario modeling, financial effects "
            "quantification, and time-horizon breakdown per ESRS E1-9."
        ),
        "category": "risk",
        "esrs_reference": "E1-9",
        "formats": ["markdown", "html", "json"],
        "version": "16.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-016 ESRS E1 Climate report templates.

    Provides centralized discovery, instantiation, and management of
    all 9 E1 climate report templates. Templates can be listed, filtered
    by category, retrieved by name, and rendered in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 9
        >>> template = registry.get("ghg_emissions_report")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}

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
            category, esrs_reference, formats, and version.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "esrs_reference": defn["esrs_reference"],
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
        return [defn["name"] for defn in TEMPLATE_CATALOG if defn["class"] is not None]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get a template instance by name.

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'ghg_emissions_report').
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
    ) -> Any:
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
            "esrs_reference": defn["esrs_reference"],
            "formats": defn["formats"],
            "version": defn["version"],
            "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string (e.g., 'emissions', 'energy',
                      'strategy', 'governance', 'actions', 'targets',
                      'credits', 'pricing', 'risk').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "esrs_reference": defn["esrs_reference"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["category"] == category and defn["class"] is not None
        ]

    def get_by_esrs_reference(self, esrs_ref: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by ESRS reference.

        Args:
            esrs_ref: ESRS reference (e.g., 'E1-6', 'E1-1').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "esrs_reference": defn["esrs_reference"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["esrs_reference"] == esrs_ref and defn["class"] is not None
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
