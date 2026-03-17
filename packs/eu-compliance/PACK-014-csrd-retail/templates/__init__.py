# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail & Consumer Goods Pack - Report Templates
================================================================

This package provides 10 retail-sector report templates for the CSRD
Retail Pack. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256
provenance hashing for audit trail integrity.

Templates:
    1. StoreEmissionsReportTemplate        - Store-level GHG emission breakdown
    2. SupplyChainReportTemplate           - Scope 3 supply chain analysis
    3. PackagingComplianceReportTemplate   - PPWR packaging compliance
    4. ProductSustainabilityReportTemplate - DPP, green claims, PEF
    5. FoodWasteReportTemplate             - Food waste management
    6. CircularEconomyReportTemplate       - MCI, take-back, EPR compliance
    7. RetailESGScorecardTemplate          - Board-level ESG scorecard (10 KPIs)
    8. ESRSRetailDisclosureTemplate        - ESRS chapter structure with audit trail
    9. ESRSRetailDisclosureReportTemplate  - Extended ESRS with materiality & audit
   10. RegulatoryDashboardTemplate         - Multi-regulation compliance dashboard

Usage:
    >>> from packs.eu_compliance.PACK_014_csrd_retail.templates import (
    ...     TemplateRegistry,
    ...     StoreEmissionsReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("store_emissions_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 14.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .store_emissions_report import StoreEmissionsReportTemplate
from .supply_chain_report import SupplyChainReportTemplate
from .packaging_compliance_report import PackagingComplianceReportTemplate
from .product_sustainability_report import ProductSustainabilityReportTemplate
from .food_waste_report import FoodWasteReportTemplate
from .circular_economy_report import CircularEconomyReportTemplate
from .retail_esg_scorecard import RetailESGScorecardTemplate
from .esrs_retail_disclosure import ESRSRetailDisclosureTemplate
from .esrs_retail_disclosure_report import ESRSRetailDisclosureReportTemplate
from .regulatory_dashboard import RegulatoryDashboardTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "StoreEmissionsReportTemplate",
    "SupplyChainReportTemplate",
    "PackagingComplianceReportTemplate",
    "ProductSustainabilityReportTemplate",
    "FoodWasteReportTemplate",
    "CircularEconomyReportTemplate",
    "RetailESGScorecardTemplate",
    "ESRSRetailDisclosureTemplate",
    "ESRSRetailDisclosureReportTemplate",
    "RegulatoryDashboardTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[StoreEmissionsReportTemplate],
    Type[SupplyChainReportTemplate],
    Type[PackagingComplianceReportTemplate],
    Type[ProductSustainabilityReportTemplate],
    Type[FoodWasteReportTemplate],
    Type[CircularEconomyReportTemplate],
    Type[RetailESGScorecardTemplate],
    Type[ESRSRetailDisclosureTemplate],
    Type[ESRSRetailDisclosureReportTemplate],
    Type[RegulatoryDashboardTemplate],
]

TemplateInstance = Union[
    StoreEmissionsReportTemplate,
    SupplyChainReportTemplate,
    PackagingComplianceReportTemplate,
    ProductSustainabilityReportTemplate,
    FoodWasteReportTemplate,
    CircularEconomyReportTemplate,
    RetailESGScorecardTemplate,
    ESRSRetailDisclosureTemplate,
    ESRSRetailDisclosureReportTemplate,
    RegulatoryDashboardTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "store_emissions_report",
        "class": StoreEmissionsReportTemplate,
        "description": (
            "Store-level GHG emissions report with Scope 1/2 breakdowns, "
            "multi-store consolidation, energy intensity KPIs, and F-Gas "
            "phase-down status tracking."
        ),
        "category": "emissions",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "supply_chain_report",
        "class": SupplyChainReportTemplate,
        "description": (
            "Scope 3 supply chain report with 15-category breakdown, "
            "hotspot analysis (top 20 suppliers/products), data quality "
            "assessment, and supplier engagement scorecard."
        ),
        "category": "supply_chain",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "packaging_compliance_report",
        "class": PackagingComplianceReportTemplate,
        "description": (
            "PPWR packaging compliance dashboard with recycled content "
            "gap analysis, EPR fee summary, eco-modulation grades, "
            "labeling compliance, and reuse progress."
        ),
        "category": "packaging",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "product_sustainability_report",
        "class": ProductSustainabilityReportTemplate,
        "description": (
            "Product sustainability report with DPP coverage, ECGT green "
            "claims audit, PEF results, textile microplastic assessment, "
            "repairability scoring, and ESPR timeline."
        ),
        "category": "product",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "food_waste_report",
        "class": FoodWasteReportTemplate,
        "description": (
            "Food waste management report with category breakdown, waste "
            "hierarchy distribution, EU 2030 30% reduction target tracking, "
            "redistribution summary, and financial impact analysis."
        ),
        "category": "waste",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "circular_economy_report",
        "class": CircularEconomyReportTemplate,
        "description": (
            "Circular economy report with Material Circularity Indicator "
            "(MCI), take-back program performance, EPR compliance by scheme, "
            "material recovery rates, and waste diversion metrics."
        ),
        "category": "circular",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "retail_esg_scorecard",
        "class": RetailESGScorecardTemplate,
        "description": (
            "Board-level ESG scorecard with 10 headline KPIs, percentile "
            "rankings, SBTi alignment, year-over-year trends, peer comparison, "
            "regulatory summary, and board-ready highlights."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "esrs_retail_disclosure",
        "class": ESRSRetailDisclosureTemplate,
        "description": (
            "ESRS retail disclosure report with chapter structure, topic-level "
            "completeness (E1, E5, S2, S4, G1), data quality notes, and "
            "audit trail references with provenance hashes."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "esrs_retail_disclosure_report",
        "class": ESRSRetailDisclosureReportTemplate,
        "description": (
            "Extended ESRS retail disclosure report with double materiality "
            "assessment, datapoint completion tracking, evidence packaging, "
            "audit readiness scoring, and cross-pack references."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
    {
        "name": "regulatory_dashboard",
        "class": RegulatoryDashboardTemplate,
        "description": (
            "Multi-regulation compliance dashboard covering CSRD, PPWR, EUDR, "
            "CSDDD, ESPR, ECGT, EU Taxonomy, and F-Gas with gap analysis, "
            "action items, timeline, and sub-sector applicability."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "14.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-014 CSRD Retail report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 retail report templates. Templates can be listed, filtered by
    category, retrieved by name, and rendered in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("store_emissions_report")
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
            name: Template name (e.g., 'store_emissions_report').
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
            category: Category string (e.g., 'emissions', 'supply_chain',
                      'packaging', 'product', 'waste', 'circular',
                      'executive', 'disclosure').

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
